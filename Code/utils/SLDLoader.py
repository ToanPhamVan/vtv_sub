import Code.SLDLoader.sld_loader as SLDLoader
import torch
import os
 # Khởi tạo đường dẫn đến thư mục chứa dữ liệu, với data_folder là thư mục chứa 

data_folder = "./skeletons/"
 # Khởi tạo class SLDLoader với dataset_path là đường dẫn đến thư mục chứa dữ liệu, 
# n_frame là số frame mỗi video, batch_size là kích thước batch, random_seed là seed 
# để shuffle dữ liệu
dataset = SLDLoader.SLD(dataset_path=data_folder,n_frame=30,batch_size=32,random_seed=42)
 #Lấy danh sách tất cả các kí hiệu
signs = os.listdir(data_folder) #[ap-dung-2294_cropped.npy,au-co
# 9908_cropped.npy,..]
 #Ví dụ muốn train kí hiệu đầu tiên trong danh sách
highlight_sign = signs[0] #ap-dung-2294_cropped.npy
# Khởi tạo DataLoader với num_data là số lượng dữ liệu muốn load, ở đây mình chọn 

data_loader = torch.utils.data.DataLoader(
dataset=dataset.get_generator(highlight_sign,num_data=128),
batch_size=32,
num_workers=0,
drop_last=True,pin_memory=True)
#Tiến hành train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(50):
    for i, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device)
        print(data.shape)
        print(label.shape)