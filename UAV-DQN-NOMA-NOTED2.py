import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import warnings
from collections import deque
from tensorflow.keras import models, layers, optimizers
import random
import copy
import matplotlib.pyplot as plt

#random seed control if necessary
#np.random.seed(1)

# set up service area range
ServiceZone_X = 500
ServiceZone_Y = 500
Hight_limit_Z = 150

# set up users' speed
MAXUserspeed = 0.5 #m/s
UAV_Speed = 5 #m/s

UserNumberPerCell = 2 # user number per UAV
NumberOfUAVs = 3 # number of UAVs
NumberOfCells = NumberOfUAVs # Each UAV is responsible for one cell
NumberOfUsers = NumberOfUAVs*UserNumberPerCell
F_c = 2 # carrier frequency/GHz
Bandwidth = 30 #khz
R_require = 0.1 # QoS data rate requirement kb
Power_level= 3 # Since DQN can only solve discrete action spaces, we set several discrete power gears, Please note that the change of power leveal will require a reset on the action space

amplification_constant = 10000 # Since the original power and noise values are sometims negligible, it may cause NAN data. We perform unified amplification to avoid data type errors
UAV_power_unit = 100 * amplification_constant # 100mW=20dBm
NoisePower = 10**(-9) * amplification_constant # noise power

class SystemModel(object):
    def __init__(
            self,
    ):
        # Initialize area
        self.Zone_border_X = ServiceZone_X
        self.Zone_border_Y = ServiceZone_Y
        self.Zone_border_Z = Hight_limit_Z

        # Initialize UAV and their location
        self.UAVspeed = UAV_Speed
        self.UAV_number = NumberOfUAVs
        self.UserperCell = UserNumberPerCell
        self.U_idx = np.arange(NumberOfUAVs) # set up serial number for UAVs
        self.PositionOfUAVs = pd.DataFrame(
           np.zeros((3,NumberOfUAVs)),
          columns=self.U_idx.tolist(),    # Data frame for saving UAVs' position
        )
        self.PositionOfUAVs.iloc[0, :] = [100, 200, 400]  # UAVs' initial x
        self.PositionOfUAVs.iloc[1, :] = [100, 400, 200]  # UAVs' initial y
        self.PositionOfUAVs.iloc[2, :] = [100, 100,100]  # UAVs' initial z

        # Initialize users and users' location
        self.User_number = NumberOfUsers
        self.K_idx = np.arange(NumberOfUsers) # set up serial number for users
        self.PositionOfUsers = pd.DataFrame(
           np.random.random((3,NumberOfUsers)),
          columns=self.K_idx.tolist(),    # Data frame for saving users' position
        )
        self.PositionOfUsers.iloc[0,:] = [204.91, 493.51, 379.41, 493.46, 258.97, 53.33] # users' initial x
        self.PositionOfUsers.iloc[1, :] = [219.75, 220.10, 49.81, 118.10, 332.59, 183.11] # users' initial y
        self.PositionOfUsers.iloc[2, :] = 0 # users' hight is assumed to be 0

        # record initial state
        self.Init_PositionOfUsers = copy.deepcopy(self.PositionOfUsers)
        self.Init_PositionOfUAVs = copy.deepcopy(self.PositionOfUAVs)

        # initialize a array to store state
        self.State = np.zeros([1, NumberOfUAVs * 3 + NumberOfUsers ], dtype=float)

        # Create a data frame for storing transmit power
        self.Power_allocation_list = pd.DataFrame(
            np.ones((1, NumberOfUsers)),
            columns=np.arange(NumberOfUsers).tolist(),
        )
        self.Power_unit = UAV_power_unit
        self.Power_allocation_list = self.Power_allocation_list * self.Power_unit

        # data frame to save distance
        self.Distence = pd.DataFrame(
            np.zeros((self.UAV_number, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # data frame to save pathloss
        self.Propergation_Loss = pd.DataFrame(
            np.zeros((self.UAV_number, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # create a data frame to save channel gain
        self.ChannelGain_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # create a data frame to save equivalent channel gain
        self.Eq_CG_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # Create a data frame to save SINR
        self.SINR_list = pd.DataFrame(
            np.zeros((1, self.User_number)),
            columns=np.arange(self.User_number).tolist(),)

        # Create a data frame to save datarate
        self.Daterate = pd.DataFrame(
        np.zeros((1, self.User_number)),
        columns=np.arange(self.User_number).tolist(),)

        # amplification_constant as mentioned above
        self.amplification_constant = amplification_constant


    def User_randomMove(self,MAXspeed,NumberofUsers):
        self.PositionOfUsers.iloc[[0,1],:] += np.random.randn(2,NumberofUsers)*MAXspeed # users random move
        return


    def Get_Distance_U2K(self,UAV_Position,User_Position,UAVsnumber,Usersnumber): # this function is for calculating the distance between users and UAVs

        for i in range(UAVsnumber):
            for j in range(Usersnumber):
                self.Distence.iloc[i,j] = np.linalg.norm(UAV_Position.iloc[:,i]-User_Position.iloc[:,j]) # calculate Distence betwen UAV i and User j

        return self.Distence


    def Get_Propergation_Loss(self,distence_U2K,UAV_Position,UAVsnumber,Usersnumber,f_c): # this function is for calculating the pathloss between users and UAVs

        for i in range(UAVsnumber):# Calculate average loss for each user,  this pathloss model is for 22.5m<h<300m d(2d)<4km
            for j in range(Usersnumber):
                UAV_Hight=UAV_Position.iloc[2,i]
                D_H = np.sqrt(np.square(distence_U2K.iloc[i,j])-np.square(UAV_Hight)) # calculate distance
                # calculate the possibility of LOS/NLOS
                d_0 = np.max([(294.05*math.log(UAV_Hight,10)-432.94),18])
                p_1 = 233.98*math.log(UAV_Hight,10) - 0.95
                if D_H <= d_0:
                    P_Los = 1.0
                else:
                    P_Los = d_0/D_H + math.exp(-(D_H/p_1)*(1-(d_0/D_H)))

                if P_Los>1:
                    P_Los = 1

                P_NLos = 1 - P_Los

                #calculate the passlos for LOS/NOLS
                L_Los = 30.9 + (22.25-0.5*math.log(UAV_Hight,10))*math.log(distence_U2K.iloc[i,j],10) + 20*math.log(f_c,10)
                L_NLos = np.max([L_Los,32.4+(43.2-7.6*math.log(UAV_Hight,10))*math.log(distence_U2K.iloc[i,j],10)+20*math.log(f_c,10)])

                Avg_Los = P_Los*L_Los + P_NLos*L_NLos # average pathloss
                gain = np.random.rayleigh(scale=1, size=None)*pow(10,(-Avg_Los/10)) # random fading
                self.Propergation_Loss.iloc[i,j] = gain #save pathloss

        return self.Propergation_Loss


    def Get_Channel_Gain_NOMA(self,UAVsnumber,Usersnumber,PropergationLosslist,UserAssociationlist,Noise_Power): # this function is for calculating channel gain

        for j in range(Usersnumber):  # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service
            i_Server_UAV = UserAssociationlist.iloc[0, j]
            Signal_power = self.amplification_constant * PropergationLosslist.iloc[i_Server_UAV, j]
            ChannelGain = Signal_power / ( Noise_Power) # calculate channel gain
            self.ChannelGain_list.iloc[0, j] = ChannelGain # save channel gain

        return self.ChannelGain_list


    def Get_Eq_CG(self,UAVsnumber,Usersnumber,PropergationLosslist,UserAssociationlist,Noise_Power): #This function is used to calculate the equivalent channel gain to determine SIC decoding order

        for j in range(Usersnumber):  # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service 'j_idx' represents the other users
            i_Server_UAV = UserAssociationlist.iloc[0, j]
            Signal_power = 100 * self.amplification_constant * PropergationLosslist.iloc[0, j] # Assuming unit power to calculate equivalent channel gain
            I_inter_cluster = 0

            for j_idx in range(Usersnumber):  # calculate Interference for user j
                if UserAssociationlist.iloc[0, j_idx] == i_Server_UAV: # if the user j_idx is user j, pass
                    pass
                else:
                    Inter_UAV = UserAssociationlist.iloc[0, j_idx]  # find the inter UAV connected with user j_idx
                    I_inter_cluster = I_inter_cluster + (
                            100 * self.amplification_constant * PropergationLosslist.iloc[Inter_UAV, j]) # calculte and add inter cluster interference

            Eq_CG = Signal_power / (I_inter_cluster + Noise_Power) # calculate equivalent channel gain for user j
            self.Eq_CG_list.iloc[0, j] = Eq_CG

        return self.Eq_CG_list


    def Get_SINR_NNOMA(self,UAVsnumber,Usersnumber,PropergationLosslist,UserAssociationlist,ChannelGain_list,Noise_Power):
        #This function is to calculate the SINR for every users

        for j in range(Usersnumber): # j represents the interfered user,  'i_Server_UAV' represents the uav providing the service 'j_idx' represents the other users
            i_Server_UAV = UserAssociationlist.iloc[0,j]
            Signal_power = self.Power_allocation_list.iloc[0,j] * PropergationLosslist.iloc[i_Server_UAV,j] # read the sinal power from power allocation list
            I_inter_cluster = 0

            for j_idx in range(Usersnumber): # calculate Interference for user j
                if UserAssociationlist.iloc[0,j_idx] == i_Server_UAV:
                    if ChannelGain_list.iloc[0,j] < ChannelGain_list.iloc[0,j_idx] and j!=j_idx: #find 'stronger' users in same cluster to count intra cluster interference
                        I_inter_cluster = I_inter_cluster + (
                                    self.Power_allocation_list.iloc[0, j_idx] * PropergationLosslist.iloc[
                                i_Server_UAV, j])  #calculate intra cluster interference

                else:
                    Inter_UAV = UserAssociationlist.iloc[0,j_idx] # calculate inter cluster interference from other UAVs
                    I_inter_cluster = I_inter_cluster + (self.Power_allocation_list.iloc[0,j_idx] * PropergationLosslist.iloc[Inter_UAV,j])#

            SINR = Signal_power/(I_inter_cluster + Noise_Power) # calculate SINR and save it
            self.SINR_list.iloc[0,j] = SINR

        return self.SINR_list


    def Calcullate_Datarate(self,SINRlist,Usersnumber,B): # calculate data rate for all users

        for j in range(Usersnumber):

            if SINRlist.iloc[0,j] <=0:
                print(SINRlist)
                warnings.warn('SINR wrong') # A data type error may occur when the data rate is too small, thus we ste up this alarm

            self.Daterate.iloc[0,j] = B*math.log((1+SINRlist.iloc[0,j]),2)

        SumDataRate = sum(self.Daterate.iloc[0,:])
        Worst_user_rate = min(self.Daterate.iloc[0,:])
        return self.Daterate,SumDataRate,Worst_user_rate


    def Reset_position(self): # save initial state for environment reset
        self.PositionOfUsers = copy.deepcopy(self.Init_PositionOfUsers)
        self.PositionOfUAVs = copy.deepcopy(self.Init_PositionOfUAVs)
        return


    def Create_state_Noposition(self,serving_UAV,User_association_list,User_Channel_Gain):
        # Create state, pay attention we need to ensure UAVs and users who are making decisions always input at the fixed neural node to achieve MDQN
        UAV_position_copy = copy.deepcopy(self.PositionOfUAVs.values)
        UAV_position_copy[:,[0,serving_UAV]] = UAV_position_copy[:,[serving_UAV,0]] # adjust the input node of serving UAV to ensure it is fixed
        User_Channel_Gain_copy = copy.deepcopy(User_Channel_Gain.values[0])

        for UAV in range(NumberOfUAVs):
            self.State[0, 3 * UAV:3 * UAV + 3] = UAV_position_copy[:, UAV].T # save UAV positions as a part of the state

        User_association_copy = copy.deepcopy(User_association_list.values)
        desirable_user = np.where(User_association_copy[0]==serving_UAV)[0] # find out the current served users

        for i in range(len(desirable_user)):
             User_Channel_Gain_copy[i],User_Channel_Gain_copy[desirable_user[i]] = User_Channel_Gain_copy[desirable_user[i]],User_Channel_Gain_copy[i] # Similarly, adjust the input node of the current served users

        for User in range(NumberOfUsers):
            self.State[0,(3*UAV+3)+User] = User_Channel_Gain_copy[User].T # save CSI of users in state

        Stat_for_return = copy.deepcopy(self.State)
        return Stat_for_return


    def take_action_NOMA(self,action_number,acting_UAV,User_asso_list,ChannelGain_list):
        UAV_move_direction = action_number % 7  #UAV has seven positional actions
        if UAV_move_direction == 0:# UAV moves along the positive half axis of the x-axis
            self.PositionOfUAVs.iloc[0,acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[0,acting_UAV] > self.Zone_border_X:
                self.PositionOfUAVs.iloc[0, acting_UAV] = self.Zone_border_X
        elif UAV_move_direction == 1: # UAV moves along the negative half axis of the x-axis
            self.PositionOfUAVs.iloc[0, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[0, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[0, acting_UAV] = 0
        elif UAV_move_direction == 2: # UAV moves along the positive half axis of the y-axis
            self.PositionOfUAVs.iloc[1, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] > self.Zone_border_Y:
                self.PositionOfUAVs.iloc[1, acting_UAV] = self.Zone_border_Y
        elif UAV_move_direction == 3: # UAV moves along the negative half axis of the y-axis
            self.PositionOfUAVs.iloc[1, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[1, acting_UAV] < 0:
                self.PositionOfUAVs.iloc[1, acting_UAV] = 0
        elif UAV_move_direction == 4: # UAV moves along the positive half axis of the z-axis
            self.PositionOfUAVs.iloc[2, acting_UAV] += self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] > self.Zone_border_Z:
                self.PositionOfUAVs.iloc[2, acting_UAV] = self.Zone_border_Z
        elif UAV_move_direction == 5: # UAV moves along the negative half axis of the z-axis
            self.PositionOfUAVs.iloc[2, acting_UAV] -= self.UAVspeed
            if self.PositionOfUAVs.iloc[2, acting_UAV] < 20:
                self.PositionOfUAVs.iloc[2, acting_UAV] = 20
        elif UAV_move_direction == 6: # UAV hold the position
            pass

        # Power allocation part - NOMA
        power_allocation_scheme = action_number//7  # decode the power allocation action,
        acting_user_list = np.where(User_asso_list.iloc[0,:] == acting_UAV)[0]
        First_user = acting_user_list[0]
        Second_user = acting_user_list[1]

        # SIC decoding order
        first_user_CG = ChannelGain_list.iloc[0,First_user]
        second_user_CG = ChannelGain_list.iloc[0,Second_user]
        if first_user_CG >= second_user_CG:
            User0 = Second_user
            User1 = First_user
        else:
            User0 = First_user
            User1 = Second_user

        # three power levels for each user
        # for the weak user, the power levels can be 2, 4, 7 * power unit
        if power_allocation_scheme % 3 == 0:
            self.Power_allocation_list.iloc[0,User0] = self.Power_unit*2
        elif power_allocation_scheme % 3 == 1:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit*4
        elif power_allocation_scheme % 3 == 2:
            self.Power_allocation_list.iloc[0, User0] = self.Power_unit*7
        # for the strong user, the power levels can be 1, 1/2, 1/4 * power unit
        if power_allocation_scheme // 3 == 0:
            self.Power_allocation_list.iloc[0,User1] = self.Power_unit
        elif power_allocation_scheme // 3 == 1:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit/2
        elif power_allocation_scheme // 3 == 2:
            self.Power_allocation_list.iloc[0, User1] = self.Power_unit/4


class DQN(object):
    def __init__(self):
        self.update_freq = 600  # Model update frequency of the target network
        self.replay_size = 10000  # replay buffer size
        self.step = 0
        self.replay_queue = deque(maxlen=self.replay_size)

        self.power_number = 3 ** UserNumberPerCell # 3 power actions
        self.action_number = 7 * self.power_number # 7 positional actions

        self.model = self.create_model() # crate model
        self.target_model = self.create_model() # crate target model

    def create_model(self):
        #Create a neural network with a input, hidden, and output layer
        STATE_DIM = NumberOfUAVs*3 + NumberOfUsers # input layer dim
        ACTION_DIM = 7 * self.power_number # output layer dim
        model = models.Sequential([
        layers.Dense(40, input_dim=STATE_DIM, activation='relu'),
        layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(learning_rate=0.001)) #Set the optimmizer and learning rate here
        return model


    def Choose_action(self, s, epsilon):
        # Choose actions according to e-greedy algorithm
        if np.random.uniform() < epsilon:
            return np.random.choice(self.action_number)
        else:
            return np.argmax(self.model.predict(s))


    def remember(self, s, a, next_s, reward):
        # save MDP transitions
        self.replay_queue.append((s, a, next_s, reward))


    def train(self,batch_size = 128, lr=1 ,factor = 1):
        if len(self.replay_queue) < self.replay_size:
            return # disable learning until buffer full
        self.step += 1

        # Over 'update_freq' steps, assign the weight of the model to the target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch) # calculate Q value
        Q_next = self.target_model.predict(next_s_batch) # predict Q value

        # update Q value following bellamen function
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        self.model.fit(s_batch, Q, verbose=0)   # DNN training


    def User_association(self,UAV_Position,User_Position,UAVsnumber, Usersnumber):
        # this function is for user association
        User_Position_array = np.zeros([Usersnumber,2])
        # convert data type
        User_Position_array[:, 0] = User_Position.iloc[0,:].T
        User_Position_array[:, 1] = User_Position.iloc[1,:].T

        K_means_association = KMeans(n_clusters=UAVsnumber).fit(User_Position_array)  # K-means approach for user association
        User_cluster = K_means_association.labels_
        Cluster_center = K_means_association.cluster_centers_

        # check if the clusters are with equal users
        for dectecter in range(UAVsnumber):
            user_numberincluster=np.where(User_cluster == dectecter)[0]
            if len(user_numberincluster) == (Usersnumber/UAVsnumber):
                 pass
            else:
                cluster_redun = []
                cluster_lack = []
                Cluster_center_of_lack=[]

                for ck_i in range(len(Cluster_center)): # Find clusters with more or less elements
                    User_for_cluster_i = np.where(User_cluster==ck_i)

                    if np.size(User_for_cluster_i) > (Usersnumber/UAVsnumber): # Find clusters with redundant elements
                        for i in range(int(np.size(User_for_cluster_i)-(Usersnumber/UAVsnumber))):
                            cluster_redun.append(ck_i)

                    if np.size(User_for_cluster_i) < (Usersnumber/UAVsnumber): # Find clusters short elements
                        for i in range(int((Usersnumber/UAVsnumber)-np.size(User_for_cluster_i))):
                            cluster_lack.append(ck_i)
                            Cluster_center_of_lack.append(Cluster_center[ck_i, :])

                # Assign redundant users to the cluster short users
                for fixer_i in range(np.size(cluster_lack)):
                    cluster_lack_fixing = cluster_lack[fixer_i]
                    Lacker_Center = Cluster_center_of_lack[fixer_i]
                    Redun_cluster = cluster_redun[fixer_i]
                    Redun_cluster_user = np.where(User_cluster==Redun_cluster) # find redundant users
                    Redun_cluster_user_postion = User_Position_array[Redun_cluster_user,:] # find redundant users' position
                    distence_U2C = np.zeros(np.size(Redun_cluster_user)) # Find the closest to the few user groups

                    for find_i in range(np.size(Redun_cluster_user)):
                        distence_U2C[find_i] = np.linalg.norm(Redun_cluster_user_postion[:,find_i]-Lacker_Center)

                    min_distence_user_order = np.where(distence_U2C==np.min(distence_U2C))
                    Redun_cluster_user_list = Redun_cluster_user[0] # Data type conversion

                    Min_d_User_idx = Redun_cluster_user_list[int(min_distence_user_order[0])]
                    User_cluster_fixed = User_cluster
                    User_cluster_fixed[Min_d_User_idx] = cluster_lack_fixing

                User_cluster = User_cluster_fixed

        if sum(User_cluster) != (UAVsnumber - 1) * Usersnumber / 2:
            warnings.warn("User association wrong")

        # Choose the nearest UAV for each user clusters
        UAV_Position_array = np.zeros([UAVsnumber, 2])
        UAV_Position_array[:, 0] = UAV_Position.iloc[0, :].T
        UAV_Position_array[:, 1] = UAV_Position.iloc[1, :].T

        User_association_list = pd.DataFrame(
            np.zeros((1, Usersnumber)),
            columns=np.arange(Usersnumber).tolist(),
        )  # data frame for saving user association indicators

        for UAV_name in range(UAVsnumber):
            distence_UAVi2C = np.zeros(UAVsnumber)
            for cluster_center_i in range(UAVsnumber):
                distence_UAVi2C[cluster_center_i] = np.linalg.norm(UAV_Position_array[UAV_name,: ] - Cluster_center[cluster_center_i])
            Servied_cluster = np.where(distence_UAVi2C==np.min(distence_UAVi2C)) # aoosciate UAV_name with the closest cluster
            Cluster_center[Servied_cluster] = 9999  # remove the selected cluster
            Servied_cluster_list = Servied_cluster[0]
            Servied_users = np.where(User_cluster==Servied_cluster_list)
            Servied_users_list = Servied_users[0]

            for i in range(np.size(Servied_users)):
                User_association_list.iloc[0,Servied_users_list[i]] = int(UAV_name) # fill UAV names in User_association_list

            User_association_list = User_association_list.astype('int') #converted data type to int

        return User_association_list


def main():
    Episodes_number = 150 # total episodes number
    Test_episodes_number = 30  # number of test episodes
    T = 60 #total time slots (steps)
    T_AS = np.arange(0, T, 200) # time solt of user association, current setting indicate 1

    env = SystemModel() # crate an environment
    agent = DQN() # crate an agent

    Epsilon = 0.9
    datarate_seq = np.zeros(T) # Initialize memory to store sum data rate
    WorstuserRate_seq = np.zeros(T) # Initialize memory to store data rate of the worst user
    Through_put_seq = np.zeros(Episodes_number) # Initialize memory to store throughput
    Worstuser_TP_seq = np.zeros(Episodes_number) # Initialize memory to store throughput of the worst user

    for episode in range(Episodes_number):
        env.Reset_position()
        #if Epsilon > 0.05: # determine the minimum Epsilon value
        Epsilon -= 0.9 / (Episodes_number - Test_episodes_number) # decaying epsilon
        p=0 # punishment counter

        for t in range(T):

            if t in T_AS:
                User_AS_List = agent.User_association(env.PositionOfUAVs, env.PositionOfUsers,NumberOfUAVs, NumberOfUsers) # user association after each period because users are moving

            for UAV in range(NumberOfUAVs):

                Distence_CG = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, NumberOfUsers) # Calculate the distance for each UAV-users
                PL_for_CG = env.Get_Propergation_Loss(Distence_CG,env.PositionOfUAVs,NumberOfUAVs, NumberOfUsers, F_c) # Calculate the pathloss for each UAV-users
                CG = env.Get_Channel_Gain_NOMA(NumberOfUAVs, NumberOfUsers, PL_for_CG, User_AS_List,NoisePower) # Calculate the channel gain for each UAV-users
                Eq_CG = env.Get_Channel_Gain_NOMA(NumberOfUAVs, NumberOfUsers, PL_for_CG, User_AS_List,NoisePower) # Calculate the equivalent channel gain to determine the decoding order

                State = env.Create_state_Noposition(UAV,User_AS_List,CG) # Generate S_t according to UAVs location and channels
                action_name = agent.Choose_action(State,Epsilon) # agent calculate action
                env.take_action_NOMA(action_name,UAV,User_AS_List,Eq_CG) # take action in the environment

                Distence = env.Get_Distance_U2K(env.PositionOfUAVs, env.PositionOfUsers, NumberOfUAVs, NumberOfUsers) # after taking actions, calculate the distance again
                P_L = env.Get_Propergation_Loss(Distence,env.PositionOfUAVs,NumberOfUAVs, NumberOfUsers, F_c) #calculate the pathloss
                SINR=env.Get_SINR_NNOMA(NumberOfUAVs,NumberOfUsers,P_L,User_AS_List,Eq_CG,NoisePower) # calculate SINR for users
                DataRate,SumRate,WorstuserRate = env.Calcullate_Datarate(SINR, NumberOfUsers, Bandwidth) # calculate data rate, sum rate and the worstusers data rate
                #print(DataRate,'\nSumrate==',SumRate,'\nWorstuserRate=',WorstuserRate)

                # calculate raward based on sum rate and check if users meet the QOS requirement
                Reward = SumRate
                if WorstuserRate < R_require:
                    Reward = Reward/2
                    p+=1

                CG_next = env.Get_Channel_Gain_NOMA(NumberOfUAVs, NumberOfUsers, P_L, User_AS_List,NoisePower)  # Calculate the equivalent channel gain for S_{t+1}
                Next_state = env.Create_state_Noposition(UAV,User_AS_List,CG_next) # Generate S_{t+1}

                #copy data for (S_t,A_t,S_t+1,R_t)
                State_for_memory = copy.deepcopy(State[0])
                Action_for_memory = copy.deepcopy(action_name)
                Next_state_for_memory = copy.deepcopy(Next_state[0])
                Reward_for_memory = copy.deepcopy(Reward)

                agent.remember(State_for_memory, Action_for_memory, Next_state_for_memory, Reward_for_memory) #save the MDP transitions as (S_t,A_t,S_t+1,R_t)
                agent.train() #train the DQN agent
                env.User_randomMove(MAXUserspeed,NumberOfUsers) # move users

                # print('UE',env.PositionOfUsers) #check user position
                # print('UAV',env.PositionOfUAVs) #check UAV position

                # save data after all UAVs moved
                if UAV==(NumberOfUAVs-1):
                    Rate_during_t = copy.deepcopy(SumRate)
                    datarate_seq[t] = Rate_during_t
                    WorstuserRate_seq[t] = WorstuserRate


        Through_put = np.sum(datarate_seq) # calculate throughput for an episode
        Worstuser_TP = np.sum(WorstuserRate_seq) # calculate throughput of the worst user for an episode
        Through_put_seq[episode] = Through_put # save throughput for an episode
        Worstuser_TP_seq[episode] = Worstuser_TP # save throughput of the worst user for an episode

        print('Episode=',episode,'Epsilon=',Epsilon,'Punishment=',p,'Through_put=',Through_put)

    # save data
    np.save("Through_put_NOMA.npy", Through_put_seq)
    np.save("WorstUser_Through_put_NOMA.npy", Worstuser_TP_seq)
    np.save("Total Data Rate_NOMA.npy", datarate_seq)
    np.save("PositionOfUsers_end_NOMA.npy",env.PositionOfUsers)
    np.save("PositionOfUAVs_end_NOMA.npy", env.PositionOfUAVs)

    # print throughput
    x_axis = range(1,Episodes_number+1)
    plt.plot(x_axis, Through_put_seq)
    plt.xlabel('Episodes')
    plt.ylabel('Throughput')
    plt.savefig('./ Throughput_NOMA.png')
    plt.show()

    # print throughput of worst users
    x_axis = range(1,Episodes_number+1)
    plt.plot(x_axis, Worstuser_TP_seq)
    plt.xlabel('Episodes')
    plt.ylabel('Throughput of Worst User')
    plt.savefig('./ WorstUser_Through_put_NOMA.png')
    plt.show()

    # print datarate of the last episode(test episode when Epsilon = 0)
    x_axis_T = range(1, T+1)
    plt.plot(x_axis_T, datarate_seq)
    plt.xlabel('Steps in test epsodes')
    plt.ylabel('Data Rate of System')
    plt.savefig('./ Total Data Rate_NOMA.png')
    plt.show()

if __name__ == '__main__':
    main()