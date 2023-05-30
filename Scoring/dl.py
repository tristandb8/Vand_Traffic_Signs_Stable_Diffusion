from torch.utils.data import Dataset
import glob
import random
import torchvision.transforms as trans
import cv2




class data_loader(Dataset):
    def __init__(self, val = False, vandTrain = False):
        # self.isTest = isTest
        self.val = val
        self.vandTrain = vandTrain
        self.convert_tensor = trans.ToTensor()
        self.resize = trans.Resize([224,224])

        # define imagenet
        imagenet = set(glob.glob("/squash/ImageNet-zstd/train/*/*"))
        print(glob.glob("/squash/ImageNet-zstd/train/*/*")[0])
        imagenet_roadsigns = set(glob.glob("/squash/ImageNet-zstd/train/n06794110/*"))
        self.imagenet = imagenet - imagenet_roadsigns
        #random.shuffle(list(self.imagenet))
        size = int(len(self.imagenet)/100)
        print("Imagenet # of images:", size)
        self.imagenet_train = set(list(self.imagenet)[:size])

        # define road signs dataset
        self.road_signs_dtst = set(glob.glob("TSRD/train/*"))
        print("road_signs_dtst # of images:", len(self.road_signs_dtst))
        
        # defined synthesized vandalized signs (for training)
        self.SV_num = 500
        self.synthesized_vand = set(glob.glob("/home/tr248228/RP_EvT/project_ml_topics/sd-vandalized-signsv2/*.png")[0:self.SV_num]) # 500 images

        # define road signs datasets - test
        self.GTSRB_test = set(glob.glob("/home/tr248228/RP_EvT/project_ml_topics/GTSRB/Test/*.png")[0:150])
        self.kaggle_test = set(glob.glob("/home/tr248228/RP_EvT/project_ml_topics/Road Sign Detection Kaggle/test/*"))
        self.TSRD_test = set(glob.glob("TSRD/test/*"))
        self.realVand_test = set(glob.glob("vandSigns/graffititrafficsigns/*"))
        print("real vand (test) # of images:", len(self.realVand_test))
        self.imagenet_test = set(list(self.imagenet)[size:int(size*1.5)])

        self.paths = set([])


        if (self.val):

            self.paths.update(self.GTSRB_test)
            self.paths.update(self.kaggle_test)
            self.paths.update(self.imagenet_test)
            self.paths.update(self.realVand_test)
            self.paths.update(self.TSRD_test)
            self.paths = list(self.paths)
            random.shuffle(self.paths)

            print("size of test", len(self.paths))
        else:


            self.paths.update(self.imagenet_train)
            self.paths.update(self.road_signs_dtst)

            # add synthesised vandalized signs in training
            if (self.vandTrain):
                # remove 500 images from dataset
                self.road_signs_dtst = set(list(self.road_signs_dtst)[self.SV_num:])
                self.paths.update(self.synthesized_vand)
            
            self.paths = list(self.paths)
            random.shuffle(self.paths)

            print("size of train", len(self.paths))


    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        # image = Image.open(self.paths[index])
        
        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
        image = self.convert_tensor(image)
        image = self.resize(image)

        if ("squash" in self.paths[index]):
            label = 0
        else:
            label =  1
        
        cat = ""
        if self.val:
            if ("ImageNet" in self.paths[index]):
                cat = "ImageNet"
            elif ("GTSRB" in self.paths[index]):
                cat = "GTSRB"
            elif ("Kaggle" in self.paths[index]):
                cat = "Kaggle"
            elif ("TSRD" in self.paths[index]):
                cat = "TSRD"
            elif ("vandSigns" in self.paths[index]):
                cat = "vandSigns"
            return image, label, cat

        return image, label

