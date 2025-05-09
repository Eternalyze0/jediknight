

# *** LIBRARIES ***


# reinforcement learning environment abstractions
import gym
# deep learning / neural network stuff for gpu
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, time, win32gui, torchvision.transforms as transforms # d
from torch.distributions import Categorical
# numerical matrix manipulation for cpu
import numpy as np
# python interfacing
import keyboard, mouse, ctypes as cts, pynput, sys
# windows interfacing
import win32gui, win32ui, win32con, win32api
from ctypes import wintypes
# ocr for on-screen momentum indicator scraping 
import easyocr
# for charts
import matplotlib.pyplot as plt
# for parsing snapshots
import json
# for image preprocessing
import cv2
# for jedi academy client interfacing
import socket


# *** HYPERPARAMETERS ***


key_possibles = ["w", "a", "s", "d", "space", "r", "e"] # legend: [forward, left, back, right, style, use, center view]
mouse_button_possibles = ["left", "middle", "right"] # legend: [attack, crouch, jump]
mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
keys_to_nums = {"w": 119, "a": 97, "s": 115, "d": 100, "space": 32, "r": 82, "e": 69}
mouse_buttons_to_nums = {"left": 141, "middle": 142, "right": 166}
n_actions = len(key_possibles)+len(mouse_button_possibles)+len(mouse_x_possibles)+len(mouse_y_possibles)
n_train_processes = 1
update_interval = 1
gamma = 0.95
max_train_ep = 100
n_filters = 2
input_rescaling_factor = 4
input_height = input_rescaling_factor * 28
input_width = input_rescaling_factor * 28
conv_output_size = 64
pooling_kernel_size = input_rescaling_factor * 2 # 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
forward_model_width = 1024
inverse_model_width = 1024
mouse_rescaling_factor = 1
dim_phi = n_actions
action_predictability_factor = 100
n_transformer_layers = 1
n_iterations = 1
inverse_model_loss_rescaling_factor = 10
jedi_momentum = 0
reward_list = []
average_reward_list = []
learning_rate_scaling_factor = 0.01 # 0.01
adam_learning_rate = 1e-7
filter_size = 10
# conv_output_ac_size = 110112
stride = 10
padding = 10
beta = 0.2
lamb = 0.1
reward_rescaling_factor = 1.0
actor_critic_reward_rescaling_factor = 100.0
actor_critic_reward_addition = 0.0 # 10 ** 10
max_forward_reward = 0.1
n_ss_fields = 8756 #4500
actor_input_layer_size = 13500
drop_rate = 0.5
nn_width = n_actions
# nn_width = 13500
baseline_reward = 0
posn_max_len = 8
total_reward = 0
n_snapshot_embedding = n_actions
sleep_time = 0.0 #0.001 # 0.00001
# n_ss_fields = nn_width
score = 0
deaths = 0
damage_dealt = 0
health = 100
shield = 25
prev_score = 0
prev_deaths = 0
prev_damage_dealt = 0
prev_health = 100
prev_shield = 25
UDP_IP = "127.0.0.1"
UDP_PORT = 29010
MESSAGE = b"3;10;0"
print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
grayscale = transforms.Grayscale(num_output_channels=1)
reader = easyocr.Reader(["en"])
rect = wintypes.RECT()
hwnd = win32gui.FindWindow(None, "OpenJK (MP)")


# *** INPUT/OUTPUT ***


def set_pos(dx, dy):
    extra = cts.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(dx, dy, 0, (0x0001), 0, cts.cast(cts.pointer(extra), cts.c_void_p))
    command=pynput._util.win32.INPUT(cts.c_ulong(0), ii_)
    cts.windll.user32.SendInput(1, cts.pointer(command), cts.sizeof(command))


def get_image(hwin=hwnd, game_resolution=(640,480), SHOW_IMAGE=False):
    image_time = time.time()
    """
    -- Inputs --

    hwin
    this is the HWND id of the cs go window
    we play in windowed rather than full screen mode
    e.g. https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getforegroundwindow

    game_resolution=(1024,768)
    is the windowed resolution of the game
    I think could get away with game_resolution=(640,480)
    and should be quicker to grab from
    but for now, during development, I like to see the game in reasonable
    size

    SHOW_IMAGE
    whether to display the image. probably a bad idea to 
    do that here except for testing
    better to use cv2.imshow("img",img) outside the funcion

    -- Outputs --
    currently this function returns img_small
    img is the raw capture image, in BGR
    img_small is a low res image, with the thought of
    using this as input to a NN

    """

    # we used to try to get the resolution automatically
    # but this didn"t seem that reliable for some reason
    # left,top,right,bottom = win32gui.GetWindowRect(hwin)
    # width = right - left 
    # height = bottom - top


    bar_height = 35 # height of header bar


    # how much of top and bottom of image to ignore
    # (reasoning we don"t need the entire screen)
    # much more efficient not to grab it in the first place
    # rather than crop out later
    # this stage can be a bit of a bottle neck
    offset_height_top = 135 
    offset_height_bottom = 135 


    offset_sides = 0 #100 # ignore this many pixels on sides, 
    width = game_resolution[0] - 2*offset_sides
    height = game_resolution[1]-offset_height_top-offset_height_bottom


    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)

    memdc.BitBlt((0, 0), (width, height), srcdc, (offset_sides, bar_height+offset_height_top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype="uint8")
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if False:
        contrast = 1.5
        brightness = 1.0
        img = cv2.addWeighted(img, contrast, img, 0, brightness)

    if SHOW_IMAGE:

        target_width = 800
        scale = target_width / img_small.shape[1] # how much to magnify
        dim = (target_width,int(img_small.shape[0] * scale))
        resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("resized",resized) 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    print("get_image() time:", time.time() - image_time)
    return img
    # return img, img_small


def snapshot_as_float_list(snapshot):
    entsize = len(snapshot["entities"][0]) # 133
    maxents = 64 #256
    nullent = [0] * entsize
    snapshot["entities"].extend([nullent] * (maxents - len(snapshot["entities"])))

    floats = []
    json_as_float_list(snapshot, floats)
    return floats


def json_as_float_list(obj, arr):
    objtype = type(obj)

    if objtype is dict:
        for x in obj.values():
            json_as_float_list(x, arr)
    elif objtype is list:
        for x in obj:
            json_as_float_list(x, arr)
    elif objtype is int:
        arr.append(float(obj))
    elif objtype is bool:
        arr.append(float(obj))
    elif objtype is float:
        arr.append(obj)


def get_snapshot(): # client snapshot
    snapshot_time = time.time()
    hwnd = win32gui.FindWindow(None, "OpenJK (MP)")
    # win32gui.SetForegroundWindow(hwnd)
    read_time = time.time()
    loop_count = 1
    while True:
        try:
            with open("C:\\Users\\yourname\\Downloads\\x86-Release\\x86-Release\\JediAcademy\\base\\player_perspective.ndjson") as f:
                last_line = json.loads(f.readline())
            break
        except:
            loop_count += 1
    while True:
        try:
            with open("C:\\Users\\yourname\\Downloads\\x86-Release\\x86-Release\\JediAcademy\\base\\action_data.ndjson", "w") as f:
                f.write("")
            break
        except:
            loop_count += 1
    print("read_time:", time.time() - read_time)
    print("loop_count", loop_count)
    global score, deaths, damage_dealt, health, shield, prev_score, prev_damage_dealt, prev_deaths, prev_health, prev_shield
    prev_score = score
    prev_deaths = deaths
    prev_damage_dealt = damage_dealt
    prev_health = health
    prev_shield = shield
    score = last_line["ps"]["persistant"][0] # score
    deaths = last_line["ps"]["persistant"][8] # deaths
    damage_dealt = last_line["ps"]["persistant"][1] # hits
    health = min(100, last_line["ps"]["stats"][0]) # health
    shield = last_line["ps"]["stats"][5] # shield
    print("score:", score) # score
    print("deaths:", deaths) # deaths
    print("damage_dealt:", damage_dealt) # damage_dealt
    print("health:", health) # health
    print("shield:", shield) # shield
    floatify = time.time()
    result = snapshot_as_float_list(last_line)
    print("floatify time:", time.time() - floatify)
    ss = torch.Tensor(result)
    ss = torch.tanh(ss)
    ss = ss.unsqueeze(dim=0)
    print("get_snapshot() time:", time.time() - snapshot_time)
    return ss


def make_plot(reward_list, average_reward_list):
    plt.plot(np.array(reward_list))
    plt.plot(np.array(average_reward_list))
    plt.title("Rewards Over Time")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.savefig("plot.png")
    plt.clf()


def release_actions():
    if "human" in sys.argv:
        return
    time_take_action = time.time()
    mouse_x = 0.0
    mouse_y = 0.0
    for i in range(len(key_possibles)):
        MESSAGE = "1;" + str(keys_to_nums[key_possibles[i]]) + ";0"
        sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
        time.sleep(sleep_time)
    for i in range(len(mouse_button_possibles)):
        MESSAGE = "1;" + str(mouse_buttons_to_nums[mouse_button_possibles[i]]) + ";0"
        sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
        time.sleep(sleep_time)
    print("mouse_dx:", mouse_x)
    print("mouse_dy:", mouse_y)
    MESSAGE = "3;" + str(int(mouse_x/mouse_rescaling_factor)) + ";" + str(int(mouse_y/mouse_rescaling_factor))
    sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
    print("release actions time:", time.time() - time_take_action)


def take_action(action):
    if "human" in sys.argv:
        return
    time_take_action = time.time()
    mouse_x = 0.0
    mouse_y = 0.0
    for i in range(len(key_possibles)):
        if action[0][i].item() == 1: # and key_possibles[i] != "s": # disable walking backwards
            # keyboard.press(key_possibles[i])
            MESSAGE = "1;" + str(keys_to_nums[key_possibles[i]]) + ";1"
            sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
        else:
            # keyboard.release(key_possibles[i])
            MESSAGE = "1;" + str(keys_to_nums[key_possibles[i]]) + ";0"
            sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
        time.sleep(sleep_time)
    for i in range(len(mouse_button_possibles)):
        if action[0][i+len(key_possibles)].item() == 1:
            # mouse.press(button=mouse_button_possibles[i])
            MESSAGE = "1;" + str(mouse_buttons_to_nums[mouse_button_possibles[i]]) + ";1"
            sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
        else:
            # mouse.release(button=mouse_button_possibles[i])
            MESSAGE = "1;" + str(mouse_buttons_to_nums[mouse_button_possibles[i]]) + ";0"
            sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
        time.sleep(sleep_time)
    for i in range(len(mouse_x_possibles)):
        if action[0][i+len(key_possibles)+len(mouse_button_possibles)].item() == 1:
            mouse_x += mouse_x_possibles[i]
    for i in range(len(mouse_y_possibles)):
        if action[0][i+len(key_possibles)+len(mouse_button_possibles)+len(mouse_x_possibles)].item() == 1:
            mouse_y += mouse_y_possibles[i]
    print("mouse_dx:", mouse_x)
    print("mouse_dy:", mouse_y)
    # set_pos(int(mouse_x/mouse_rescaling_factor), int(mouse_y/mouse_rescaling_factor))
    MESSAGE = "3;" + str(int(mouse_x/mouse_rescaling_factor)) + ";" + str(int(mouse_y/mouse_rescaling_factor))
    sock.sendto(str.encode(MESSAGE), (UDP_IP, UDP_PORT))
    print("take action time:", time.time() - time_take_action)


# ENVIRONMENT
# the custom gym environment, agent <-> environment is the canonical reinforcement learning loop, here curiosity, since it provides rewards, is part of the environment for code simplicity


def wake_sleep(period): # for oscillating preferences eg exploratory <-> exploitative
    global n_iterations
    return np.cos(2 * np.pi * n_iterations/period)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.average_state = 0
        self.n_states = 0
        img = torch.stack([snapshot_model(get_snapshot().float().to(device)), image_model(torch.Tensor(np.array(get_image())).unsqueeze(0).float().to(device)).unsqueeze(0)])
        frame1 = torch.tensor(img)
        frame2 = torch.tensor(img)
        self.average_frame = frame2
        state = torch.cat([frame1, frame2], dim=0)
        self.previous_frame = frame2
        self.previous_state = state
        self.n_frames_seen = 1
        self.average_reward = 0
    def step(self, action):
        global reward_list, average_reward_list, n_iterations, total_reward
        take_action(action)
        img = torch.stack([snapshot_model(get_snapshot().float().to(device)), image_model(torch.Tensor(np.array(get_image())).unsqueeze(0).float().to(device)).unsqueeze(0)])
        frame1 = self.previous_frame
        # frame2 = torch.tensor(np.array(img)).float().float().to(device)
        frame2 = img
        self.average_frame = (self.n_frames_seen * self.average_frame + frame2) / (self.n_frames_seen + 1)
        self.n_frames_seen += 1 
        # frame2[0][0][0] = n_iterations
        state = torch.cat([frame1, frame2], dim=0)
        phi_previous_state = phi_model(self.previous_state)
        phi_state = phi_model(state)
        phi_previous_state = phi_previous_state.unsqueeze(dim=0)
        phi_state = phi_state.unsqueeze(dim=0)
        print(phi_previous_state.shape)
        print(phi_state.shape)
        action_hat = inverse_model(torch.cat([phi_previous_state, phi_state], dim=1))
        # action = torch.unsqueeze(action, dim=0)
        error_inverse_model = (1-beta) * F.nll_loss(action_hat.permute(0, 2, 1), action.long(), size_average=None, reduce=None, reduction="mean") # input, target
        print("error_inverse_model:", error_inverse_model.item())
        optimizer_inverse.zero_grad()
        error_inverse_model.backward(retain_graph=True)
        if "sign" in sys.argv:
            for p in list(inverse_model.parameters())+list(phi_model.parameters()):
                p.grad = torch.sign(p.grad)
        optimizer_inverse.step()
        phi_previous_state_f = phi_model(self.previous_state)
        action_f = action.clone()
        phi_state_f = phi_model(state)
        # phi_previous_state_f = phi_previous_state_f.unsqueeze(0)
        print(action_f.shape)
        print(phi_previous_state_f.shape)
        # phi_previous_state_f = phi_previous_state_f.squeeze(dim=0)
        phi_hat_state_f = forward_model(torch.cat([action_f, phi_previous_state_f], dim=1))
        # phi_state_f = phi_state_f.unsqueeze(0)
        phi_hat_state_f = phi_hat_state_f.squeeze(0)
        phi_state_f = phi_state_f.squeeze(0)
        print(phi_hat_state_f.shape)
        print(phi_state_f.shape)
        error_forward_model = beta * torch.nn.functional.mse_loss(phi_hat_state_f, phi_state_f, size_average=None, reduce=None, reduction="mean") # input, target
        print("error_forward_model:", error_forward_model.item())
        optimizer_forward.zero_grad()
        error_forward_model.backward(retain_graph=True)
        if "sign" in sys.argv:
            for p in forward_model.parameters():
                p.grad = torch.sign(p.grad)
        optimizer_forward.step()
        print("reward components:", error_forward_model.item(), error_inverse_model.item(), jedi_momentum)
        # reward = wake_sleep(period=100) * (-0.001) * ( 100 * error_forward_model + error_inverse_model ) + jedi_momentum + 1000
        r_curiosity = 100 * error_forward_model - 100 * error_inverse_model                                                # curiosity
        r_momentum  = 0 * jedi_momentum                                                                                      # momentum
        r_damage    = ( damage_dealt - prev_damage_dealt ) + 0.5 * ( health - prev_health ) + 0.5 * ( shield - prev_shield ) # damage
        r_score     = 100 * ( ( score - prev_score ) - 0.5 * ( deaths - prev_deaths ) )                                      # score
        r_baseline  = 1800                                                                                                    # baseline
        reward = r_curiosity + r_momentum + r_damage + r_score + r_baseline
        print("REWARDS")
        print("r_curiosity:", r_curiosity)
        print("r_momentum:", r_momentum)
        print("r_damage:", r_damage)
        print("r_score:", r_score)
        print("r_baseline:", r_baseline)
        # print("wake sleep", wake_sleep(period=1)) # ~ 25 s
        print("reward:", reward)
        # reward_list.append(reward)
        total_reward += reward
        print("average reward:", total_reward/n_iterations)
        # self.average_reward = ( self.average_reward * self.n_frames_seen + jedi_momentum ) / (self.n_frames_seen + 1)
        # print("average momentum:", self.average_reward)
        # average_reward_list.append(self.average_reward)
        done = False
        if ( n_iterations % 100 ) == 0:
            done = True
            release_actions()
            print("---------------------------------------------------------- EPISODE "+str(n_iterations//100)+" COMPLETE")
        info = {}
        self.previous_state = state
        self.previous_frame = frame2
        return state, reward, done, info
    def reset(self):
        img = torch.stack([snapshot_model(get_snapshot().float().to(device)), image_model(torch.Tensor(np.array(get_image())).unsqueeze(0).float().to(device)).unsqueeze(0)])
        frame1 = self.previous_frame
        frame2 = torch.tensor(img)
        state = torch.cat([frame1, frame2], dim=0)
        self.average_state = state
        self.n_states = 1
        return state
    def render(self, mode="human"):
        pass
    def close(self):
        pass


# *** NEURAL MODELS ***


class SnapshotModel(nn.Module): # map snapshot to a preferred size
    def __init__(self):
        super(SnapshotModel, self).__init__()
        self.fc1 = nn.Linear(n_ss_fields, n_snapshot_embedding)

    def forward(self, x):
        x = self.fc1(x)
        return x


class ImageModel(nn.Module): # exploit spatial symmetry of images and map to a preferred size
    def __init__(self):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, filter_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(n_filters, n_filters, filter_size, stride=stride, padding=padding)
        self.fc1 = nn.Linear(conv_output_size, n_snapshot_embedding)
    def forward(self, x):
        x = grayscale(x)
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = F.elu(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        return x


class PositionalEncoding(nn.Module): # encode real numbers using sines and cosines in the style of transformers
    def __init__(self, d_model: int, dropout: float = drop_rate, max_len: int = posn_max_len): # 5000
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.Tensor([10000.0])) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ActorCritic(nn.Module): # the neural agent reinforcement learning model = f: image -> action
    def __init__(self):
        super(ActorCritic, self).__init__()
        # self.posn = PositionalEncoding(n_ss_fields * 4, dropout=0.1, max_len=posn_max_len)
        self.fc1 = nn.Linear(n_actions * 4, nn_width)
        self.fc2 = nn.Linear(nn_width, nn_width)
        self.dropout = nn.Dropout(drop_rate)
        self.fc_pi = nn.Linear(nn_width, n_actions * 2)
        self.fc_v = nn.Linear(nn_width, 1)
    def pi(self, x, softmax_dim=-1):
        x = x.reshape(1, n_actions * 4)
        # x = self.posn(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc_pi(x)
        x = x.reshape(1, n_actions, 2)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    def v(self, x): # value (like Q-value in Q-learning, an estimate of the expected cumulative future rewards)
        x = x.reshape(1, n_actions * 4)
        # x = self.posn(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc_v(x)
        return x


class PhiModel(nn.Module): # the neural state embedding model = f: image -> compressed representation
    def __init__(self):
        super(PhiModel, self).__init__()
        # self.posn = PositionalEncoding(n_ss_fields * 4, dropout=0.1, max_len=posn_max_len)
        self.fc1 = nn.Linear(n_actions * 4, nn_width)
        self.fc2 = nn.Linear(nn_width, dim_phi)
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        x = x.reshape(1, n_actions * 4)
        # x = self.posn(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ForwardModel(nn.Module): # the neural future prediction model = f: compressed represenation at time t -> compressed representation at time t+1
    def __init__(self):
        super(ForwardModel, self).__init__()
        # self.posn = PositionalEncoding(n_actions+dim_phi)
        self.fc1 = nn.Linear(n_actions+dim_phi, nn_width)
        self.fc2 = nn.Linear(nn_width, dim_phi)
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        # x = self.posn(x)
        # x_init = x
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.fc4(x) #+ torch.narrow(input=x_init, dim=1, start=n_actions, length=dim_phi) # skip connection so forward doesn't have to remember t when computing t+1
        return x
    

class InverseModel(nn.Module): # the neural inverse dynamics model = f: compressed state at time t, compressed state at time t+1 -> action
    def __init__(self):
        super(InverseModel, self).__init__()
        # self.posn = PositionalEncoding(dim_phi * 2)
        self.fc1 = nn.Linear(dim_phi * 2, nn_width)
        self.fc2 = nn.Linear(nn_width, n_actions * 2)
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        x = x.reshape(1, dim_phi * 2)
        # x = self.posn(x)
        x = self.fc1(x) #+ torch.narrow(input=x_init, dim=1, start=n_actions, length=dim_phi)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.reshape(1, n_actions, 2)
        prob = F.log_softmax(x, dim=2)
        return prob


# *** PLAY AND LEARN ***


def train():
    global n_iterations
    local_model = ActorCritic().float().to(device)
    local_model.load_state_dict(global_model.state_dict())
    env = CustomEnv()
    start_time = time.time()
    for n_epi in range(max_train_ep):
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                n_iterations += 1
                print("--------- n_iterations:", n_iterations)
                print("fps:", n_iterations / (time.time() - start_time))
                if (n_iterations % 1000) == 0 and not "nosave" in sys.argv:
                    print("saving model..")
                    torch.save(global_model, "jedi_actorcritic_model.pth")
                    torch.save(phi_model, "jedi_embedding_model.pth")
                    torch.save(inverse_model, "jedi_inverse_model.pth")
                    torch.save(forward_model, "jedi_forward_model.pth")
                    torch.save(image_model, "jedi_image_model.pth")
                    torch.save(snapshot_model, "jedi_snapshot_model.pth")
                    print("model saved.")
                if "plot" in sys.argv and (n_iterations % 1000) == 0:
                    make_plot(reward_list, average_reward_list)
                prob = local_model.pi(s.float())
                m = Categorical(prob)
                # a = m.sample().item()
                a = m.sample().float().to(device)
                s_prime, r, done, info = env.step(a)
                s_lst.append(s)
                # a_lst.append([a])
                a_lst.append(a)
                # r_lst.append(r/100.0)
                r_lst.append(r/actor_critic_reward_rescaling_factor + actor_critic_reward_addition)
                s = s_prime
                if done:
                    break
            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final.float()).item()
            print("R:", R)
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                # td_target_lst.append([R])
                td_target_lst.append(R)
            td_target_lst.reverse()

            # s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            #     torch.tensor(td_target_lst)
            s_batch = torch.stack(s_lst, dim=0)
            a_batch = torch.stack(a_lst, dim=0)
            td_target = torch.stack(td_target_lst, dim=0)
            advantage = td_target - local_model.v(s_batch)

            # pi = local_model.pi(s_batch, softmax_dim=1)
            pi = local_model.pi(s_batch)
            pi = pi.permute(0, 2, 1)
            pi_a = pi.gather(1, a_batch.type(torch.LongTensor).to(device))
            # loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())
            loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.v(s_batch).squeeze(0).squeeze(0), td_target.detach())

            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
                # loss.mean().backward(retain_graph=True)
            loss.mean().backward(retain_graph=True)
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

    env.close()
    print("Training process reached maximum episode.")
    

# *** INITIALIZE AND RUN TRAIN ***


if __name__ == "__main__":
    while True:
        n_iterations = 0
        try:
            del global_model
            del phi_model
            del inverse_model
            del forward_model
            del image_model
            del snapshot_model
        except:
            pass
        if "check" in sys.argv:
            reward_list = np.load("reward_list.npy")
            print(reward_list)
            print(max(reward_list))
            sys.exit()
        global_model = ActorCritic().float().to(device)
        phi_model = PhiModel().float().to(device)
        inverse_model = InverseModel().float().to(device)
        forward_model = ForwardModel().float().to(device)
        image_model = ImageModel().float().to(device)
        snapshot_model = SnapshotModel().float().to(device)
        if not "new" in sys.argv:
            print("loading model..")
            global_model = torch.load("jedi_actorcritic_model.pth", weights_only=False).float()
            phi_model = torch.load("jedi_embedding_model.pth", weights_only=False).float()
            inverse_model = torch.load("jedi_inverse_model.pth", weights_only=False).float()
            forward_model = torch.load("jedi_forward_model.pth", weights_only=False).float()
            image_model = torch.load("jedi_image_model.pth", weights_only=False).float()
            snapshot_model = torch.load("jedi_snapshot_model.pth", weights_only=False).float()
            print("model loaded.")
        trainable_actorcritic_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
        print(f"Total number of trainable actor-critic model parameters: {trainable_actorcritic_params}")
        trainable_inverse_params = sum(p.numel() for p in list(phi_model.parameters())+list(inverse_model.parameters()) if p.requires_grad)
        print(f"Total number of trainable inverse model parameters: {trainable_inverse_params}")
        trainable_forward_params = sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
        print(f"Total number of trainable forward model parameters: {trainable_forward_params}")
        if "show" in sys.argv:
            sys.exit()
        if "sign" in sys.argv:
            print("using sign gradient descent.")
            optimizer = optim.SGD(global_model.parameters(), lr=learning_rate_scaling_factor*1.0/float(trainable_actorcritic_params))
            optimizer_inverse = optim.SGD(list(inverse_model.parameters()) + list(phi_model.parameters()), lr=learning_rate_scaling_factor*1.0/float(trainable_inverse_params))
            optimizer_forward = optim.SGD(forward_model.parameters(), lr=learning_rate_scaling_factor*1.0/float(trainable_forward_params))
        else:
            optimizer = optim.AdamW(global_model.parameters(), lr=adam_learning_rate) # 0.00001
            optimizer_inverse = optim.AdamW(list(inverse_model.parameters()) + list(phi_model.parameters()), lr=adam_learning_rate) #lr=0.01)
            optimizer_forward = optim.AdamW(forward_model.parameters(), lr=adam_learning_rate) #lr=0.01)
        global_model.share_memory()
        train()
    sys.exit()
