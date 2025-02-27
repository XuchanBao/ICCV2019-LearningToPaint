import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
from DRL.ddpg import decode
from Renderer.quadratic_gen import get_initialization, generate_quadratic_heatmap
from utils.util import *
from PIL import Image
from torchvision import transforms, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aug = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             ])

width = 128
convas_area = width * width

img_train = []
img_test = []
train_num = 0
test_num = 0

class Paint:
    def __init__(self, obs_dim, act_dim, batch_size, max_step):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = act_dim
        self.observation_space = (self.batch_size, width, width, obs_dim)
        self.test = False

        self.parameters = None
        
    def load_data(self):
        # CelebA
        global train_num, test_num
        for i in range(200000):
            img_id = '%06d' % (i + 1)
            try:
                img = cv2.imread('../data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (width, width))
                if i > 2000:
                    train_num += 1
                    img_train.append(img)
                else:
                    test_num += 1
                    img_test.append(img)
            finally:
                if (i + 1) % 10000 == 0:
                    print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(train_num), str(test_num)))
        
    def pre_data(self, id, test):
        if test:
            img = img_test[id]
        else:
            img = img_train[id]
        if not test:
            img = aug(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))
    
    def reset(self, test=False, begin_num=False):
        self.test = test
        self.imgid = [0] * self.batch_size
        self.gt = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num)  % test_num
            else:
                id = np.random.randint(train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self.pre_data(id, test))
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0

        self.parameters = get_initialization(self.batch_size)
        self.canvas, _, __ = generate_quadratic_heatmap(self.parameters)
        self.canvas = self.canvas.reshape(-1, 3, width, width).to(device)

        self.parameters = self.parameters.unsqueeze(-1).unsqueeze(-1).expand(self.parameters.size() + (128, 128))
        self.parameters = self.parameters.to(device)

        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # params B * 2 * width * width
        # T B * 1 * width * width
        # Total: (B, 9, width, width)
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas.float(), self.gt.float(), self.parameters, T.float().to(device)), 1) # canvas, gt_img, params, T

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action):
        self.canvas, self.parameters = decode(action, self.parameters)
        self.canvas = (self.canvas * 255).byte()

        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)
