from PIL import Image,ImageOps
import os
import os.path
import numpy as np
import pickle
#import cv2
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms



class LyftLoader():

    def __init__(self, root, img_count=None, train=True, transform=None, bsize=None):

        #super( self).__init__(root, transform=transform,target_transform=target_transform)
        self.img_count = img_count
        self.bsize = bsize
        self.train = train  # training set or test set
        if self.train:
            self.Input_images,self.Input_len = self.LoadPairs_corrected(root)
            self.path_run = 'training'
        else:
            self.Input_images,self.Input_len = self.LoadPairs_corrected_test(root)
            self.path_run = 'testing'
            #self.path_run = 'testing'

        self.transform = transform
        self.root = root
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            ])

        self.target_transform = target_transform

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        # for file_name, checksum in downloaded_list:
        #     file_path = os.path.join(self.root, self.base_folder, file_name)
        #     with open(file_path, 'rb') as f:
        #         entry = pickle.load(f, encoding='latin1')
        #         self.data.append(entry['data'])
        #         if 'labels' in entry:
        #             self.targets.extend(entry['labels'])
        #         else:
        #             self.targets.extend(entry['fine_labels'])
        #
        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC



    def LoadPairs_corrected(self,root):

        InputImage = []

        #batches = sorted(os.listdir(BatchLocation + '/'))
        #batches = sorted(os.listdir(root + '/'),key=lambda x: int(os.path.splitext(x)[0]))
        batches = sorted(os.listdir(root + '/raw/'))

        if self.img_count==0:
            bacthes_size = len(batches)
        else:
            bacthes_size = self.img_count

        # to put it normaly


        for b in range(0, bacthes_size):
            batchName = batches[b]
            InputImage.append(batchName)

        return InputImage,len(InputImage)


    def LoadPairs_corrected_test(self,root):

        InputImage = []

        #batches = sorted(os.listdir(BatchLocation + '/'))
        #batches = sorted(os.listdir(root + '/'),key=lambda x: int(os.path.splitext(x)[0]))
        batches = sorted(os.listdir(root + '/'))
        bacthes_size = 1

        # to put it normaly


        for b in range(0, bacthes_size):
            batchName = batches[b]
            InputImage.append(batchName)

        return InputImage,len(InputImage)



    def __len__(self):
        return len(self.Input_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       # img, target = self.data[index], self.targets[index]


        #img_p = '/home/data3/hichem/DLP_Project/Data/Lyft/raw/' + self.Input_images[index][:-3]+'jpeg'

        #camId = self.Input_images[index][13]
        #pathImages = '/home/data3/hichem/Lyft_Level_5/v1.02-train/images/cam' + camId + '/'

        #img_p = pathImages + self.Input_images[index][:-3]+'jpeg'
        #img_p = '/home/data3/hichem/DLP_Project/Data/Lyft_final/training/raw/' + self.Input_images[index][:-3]+'jpeg'
        img_p = self.root +'raw/' + self.Input_images[index]
        target_p = self.root +'bin/' + self.Input_images[index]

        img = PIL.Image.open(img_p)
        #img = ImageOps.grayscale(img)

        #img_np = np.array(img)

        target = PIL.Image.open(target_p)

        img_out = self.transform(img)
        target_out = self.target_transform(target)
        # plt.imshow(np.transpose(target_out[:,:,:].cpu().numpy(), axes=[1,2,0]))


        return img_out, target_out, self.Input_images[index][:-4]





    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


