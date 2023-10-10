import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorc1h_toolbelt.modules import DropPath
import timm
from timm.models.layers import DropPath
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from perlin_numpy import generate_fractal_noise_3d


# nn.Linear nn.Conv3d nn.Conv2d nn.ConvTransposed3d nn.ConvTransposed2d

# nn.GELU nn.Dropout DropPath nn.LayerNorm F.softmax

# roll, nn.*pad/F.pad F.crop(2d) 实际上可以直接对 tensor 使用切片操作，参考下文
# https://stackoverflow.com/questions/57517121/how-can-i-do-the-centercrop-of-3d-volumes-inside-the-network-model-with-pytorch

# tensor.reshape tensor.permute 

# torch.tensor torch.normal_ torch.arange+torch.reshape/view

# torch.linspace torch.meshgrid torch.stack torch.flatten sum abs concatenate

# load backward optim.Adam save

# load ERA5 masks mean std

def LoadModel(path):
    '''
    load pretrained model
    path: model_path
    '''
    return onnx.load('pangu_weather_24.onnx')

def DownloadData():
    '''
    download train data from cds
    write in data.nc
    '''

    import cdsapi

    # 创建CDS API客户端
    c = cdsapi.Client()

    # 定义ERA5数据请求参数
    data_request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'your_variable_name',  # 指定感兴趣的气象变量
        'year': '2023',
        'month': '01',
        'day': '01',
        'time': ['00:00', '01:00', '02:00'],  # 时间步长
        'area': [90, -180, -90, 180],  # 区域范围（北极到南极，全球范围）
        'grid': [0.25, 0.25],  # 空间分辨率
        'target': 'data.nc',  # 输出文件名
    }

    # 使用CDS API执行数据请求
    c.retrieve('reanalysis-era5-single-levels', data_request)

def LoadStatic(data_dir):
    '''
    load data from local file
    return weather_mean, weather_std, weather_surface_mean, weather_surface_std
    '''
    # with nc.Dataset(data_dir) as nc_file:
        
    #     data = nc_file.variables['var']

    # FIXME 此处返回的均值和标准差是对某一个变量的统计量
    return weather_mean, weather_std, weather_surface_mean, weather_surface_std

def Inference(input, input_surface, forecast_range):
    '''
    input: input tensor, need to be normalized to N(0, 1) in practice
    input_surface: target tensor, need to be normalized to N(0, 1) in practice
    forecast_range: iteration numbers when roll out the forecast model
    '''
    # 加载预训练模型
    ModelPath24 = ""
    ModelPath6 = ""
    ModelPath3 = ""
    ModelPath1 = ""
    PanguModel24 = LoadModel(ModelPath24)
    PanguModel6 = LoadModel(ModelPath6)
    PanguModel3 = LoadModel(ModelPath3)
    PanguModel1 = LoadModel(ModelPath1)

    # 加载静态数据，获取均值和标准差
    weather_mean, weather_std, weather_surface_mean, weather_surface_std = LoadStatic()

    # 使用初始数据将每个模型的输入初始化
    # input 和 input_surface 会作为临时变量多次复用
    input_24, input_surface_24 = input, input_surface
    input_6, input_surface_6 = input, input_surface
    input_3, input_surface_3 = input, input_surface

    # 使用 list 存储输出数据
    output_list = []

    # 对 [1,forecast_range] hour 范围做预测
    for i in range(forecast_range):
        # 如果时长是 24 的倍数则使用 24h 模型
        if (i+1) % 24 == 0:
            # 使用 24 数据
            input, input_surface = input_24, input_surface_24

            # 调用 24h 模型
            output, output_surface = PanguModel24(input, input_surface)

            # 归一化的逆向过程，即从归一化的数据恢复原始数据
            output = output * weather_std + weather_mean
            output_surface = output_surface * weather_surface_std + weather_surface_mean

            # 存储预测结果用于下轮预测
            input_24, input_surface_24 = output, output_surface
            input_6, input_surface_6 = output, output_surface
            input_3, input_surface_3 = output, output_surface

        # switch to the 6-hour model if the forecast time is 30 hours, 36 hours, ..., 24*N + 6/12/18 hours
        elif (i+1) % 6 == 0:
            # Switch the input back to the stored input
            input, input_surface = input_6, input_surface_6

            # Call the model pretrained for 6 hours forecast
            output, output_surface = PanguModel6(input, input_surface)

            # Restore from uniformed output
            output = output * weather_std + weather_mean
            output_surface = output_surface * weather_surface_std + weather_surface_mean
            
            # Stored the output for next round forecast
            input_6, input_surface_6 = output, output_surface
            input_3, input_surface_3 = output, output_surface

            # switch to the 3-hour model if the forecast time is 3 hours, 9 hours, ..., 6*N + 3 hours
        elif (i+1) % 3 ==0:
            # Switch the input back to the stored input
            input, input_surface = input_3, input_surface_3

            # Call the model pretrained for 3 hours forecast
            output, output_surface = PanguModel3(input, input_surface)

            # Restore from uniformed output
            output = output * weather_std + weather_mean
            output_surface = output_surface * weather_surface_std + weather_surface_mean
            
            # Stored the output for next round forecast
            input_3, input_surface_3 = output, output_surface

        # switch to the 1-hour model
        else:
            # Call the model pretrained for 1 hours forecast
            output, output_surface = PanguModel1(input, input_surface)

            # Restore from uniformed output
            output = output * weather_std + weather_mean
            output_surface = output_surface * weather_surface_std + weather_surface_mean

            # Stored the output for next round forecast
            input, input_surface = output, output_surface

        # Save the output
        output_list.append((output, output_surface))
    return output_list

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, dim):
        super(PatchEmbedding, self).__init__()
        # 两个卷积层的目的是将数据转为立方体型
        # conv3d 卷积层，输入通道 5，输出通道 dim，卷积核大小 patch_size，步长 patch_size
        self.conv = nn.Conv3d(in_channels=5, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        # conv2d 卷积层，输入通道 7，输出通道 dim，卷积核大小 patch_size[1:]，步长 patch_size[1:]
        self.conv_surface = nn.Conv2d(in_channels=7, out_channels=dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        # 加载常量掩码
        self.land_mask, self.soil_type, self.topography = LoadConstantMask()
        # 计算不变填充大小
        self.padding_depth = ((patch_size[0] - 1) * patch_size[0]) / 2
        self.padding_height = ((patch_size[1] - 1) * patch_size[1]) / 2
        self.padding_width = ((patch_size[2] - 1) * patch_size[2]) / 2

    def forward(self, input, input_surface):
        # 注意 pad 的输入参数格式
        # input 表示要填充的数据，其维度表示为（patch_size, channels, **dims）
        # 即第一个数字表示批次大小，第二数字表示通道数，后面的数字表示维度
        # 例如 torch.empty(1,1,2,3,4) 表示批次大小为 1，通道数 1，维度为 (2,3,4) 的输入数据
        # pad 参数接受一个元组，这个元组的元素个数 m 必须满足 m/2 <= input_dims 且 m%2=0
        # pad 的参数具有固定的含义，从前往后两两一堆，分别表示在 0 轴前后的填充，1 轴前后的填充，以此类推。
        # 例如 pad = (1,2,3,4)  表示在 0 轴前填充 1 单位，0 轴后填充 2  单位，1 轴前填充 3 单位，1 轴后填充 4 单位
        padding_3d = (self.padding_width, self.padding_width, self.padding_height, self.padding_height, self.padding_depth, self.padding_depth)
        padding_2d =self.padding_width, self.padding_width, self.padding_height, self.padding_height()
        # FIXME 此处应该可以直接在 Conv 卷积层做 padding 处理，而不需要单独处理输入数据
        input = F.pad(input=input, pad=padding_3d) # input 为 3d 数据，做 3d 的 0 填充
        input_surface = F.pad(input=input_surface, pad=padding_2d) # 2d 填充
        # 卷积
        input = self.conv(input)
        # 表层数据合并
        input_surface = torch.concat(input_surface, self.land_mask, self.soil_type, self.topography)
        # 卷积
        input_surface = self.conv_surface(input_surface)

        # 将表层数据和高空压力场合并
        x = torch.concat(input, input_surface)
        # 修改维度顺序
        x = torch.permute(x, (0, 2, 3, 4, 1))
        # reshape
        x = torch.reshape(x, (x.shape[0], 8*360*181, x.shape[-1]))
        return x


class Mlp(nn.Module):
    def __init__(self, dim, dropout_rate):
        super(Mlp, self).__init__()
        '''
        MLP 层，类似于常见的 vision transformer 层结构
        '''
        self.linear1 = nn.Linear(dim, dim*4)
        self.linear2 = nn.Linear(dim*4, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x
    
    def PerlinNoise():
        '''
        随机柏林噪声
        '''
        # Define number of noise
        octaves = 3
        # Define the scaling factor of noise
        noise_scale = 0.2
        # Define the number of periods of noise along the axis
        period_number = 12
        # The size of an input slice
        H, W = 721, 1440
        # Scaling factor between two octaves
        persistence = 0.5
        # see https://github.com/pvigier/perlin-numpy/ for the implementation of GenerateFractalNoise (e.g., from perlin_numpy import generate_fractal_noise_3d)
        # 使用 perlin_numpy 实现的 generate_fractal_noise_3d 方法
        perlin_noise = noise_scale*np.generate_fractal_noise_3d((H, W), (period_number, period_number), octaves, persistence)
        return perlin_noise

class EarthAttention3D(nn.Module):
    def __init__(self, dim, heads, dropout_rate, window_size) -> None:
        '''
        3d window attention with Earth-Specific bias
        '''
        super(EarthAttention3D, self).__init__()
        # 该线性层在伪代码中使用的参数为 (dim, dim=3)，经查看 https://github.com/microsoft/Swin-Transformer 认为此处为笔误，应该是 (dim, dim*3)
        self.linear1 = nn.Linear(dim, dim*3, bias=True)
        self.linear2 = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        self.head_number = heads
        self.dim = dim
        self.scale = (dim//heads)**-0.5
        self.window_size = window_size

        # FIXME 此处的 input_sahpe 是 self.forward 方法的输入数据的 shape，请运行后记录
        self.type_of_windows = (input_shape[0]//window_size[0]) * (input_shape[1]//window_size[1])
        # 对于不同的 type_of_windows，根据 paper 构建参数集合，并将其转为可学习参数
        shape=((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, heads)
        self.earth_specific_bias = nn.Parameter(torch.empty(shape))
        # 对 self.earth_specific_bias 做正态分布截断，std=0.02
        self.earth_specific_bias = nn.init.trunc_normal_(self.earth_specific_bias, std=0.02)
        # 创建索引以重用 self.earth_specific_bias
        self.position_index = self._construct_index()

    def _construct_index(self):
        '''
        构建重用对称变量的位置索引？
        '''
        # 查询矩阵中的压力等级索引
        coords_zi = torch.range(self.window_size[0])
        # 键矩阵中的压力等级索引
        coords_zj = -1 * torch.range(self.window_size[0]) * self.window_size[0 ]

        # 查询矩阵中的维度索引
        coords_hi = torch.range(self.window_size[1])
        # 键矩阵中的维度索引
        coords_hj = -1 * torch.range(self.window_size[1]) * self.window_size[1]

        # 键值对的经度索引
        coords_w = torch.range(self.window_size[2])

        # 修改索引顺序，用于总体计算
        # stack 方法用于在指定维度上堆叠 tensor
        coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
        coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
        coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
        coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
        coords = coords_flatten_1[:,:,None] - coords_flatten_2[:,None,:]
        coords = torch.permute(coords, (1, 2, 0))

        # 将各个维度的索引转为从 0 开始
        coords[:,:,2] += self.window_size[2]-1
        coords[:,:,1] *= 2 * self.window_size[2]-1
        coords[:,:,0] *= (2 * self.window_size[2] - 1) * self.window_size[1] * self.window_size[1]

        # FIXME dim=-1? 对索引求和
        self.position_index = torch.sum(coords, dim=-1)

        # 展平索引
        self.position_index = torch.flatten(self.position_index)

    def forward(self, x, mask):
        # 线性层：创建查询、键、值，此处将维度扩展三倍
        x = self.linear1(x)

        # 记录 input 的原始 shape
        original_shape = x.shape

        # 重塑数据来计算多头注意力
        # 此处各个维度的含义为（batch_size, sequence_size, 3, head_n, head_dim）
        # 其中第三个维度的值为 3 表示 query, key, value 三个矩阵，后续会进行分离
        qkv = torch.reshape(x, shape=(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number))
        # q k v 分离
        # 重新排列维度，将 3 提升至第一位，然后进行分离
        qkv = torch.permute(qkv, (2, 0, 3, 1, 4))
        query = qkv[0]
        key = qkv[1]
        value = qkv[2]

        # 缩放 q 矩阵
        query = query * self.scale

        # 计算 attention，添加一个可学习的偏置用于修复网格的非一致性
        # q = (x.shape[0], head_number, x.shape[1], head_dim)
        # k = q
        # attention = q * k.T = (x.shape[0], head_number, x.shape[1], x.shape[1])
        attention = torch.matmul(query, torch.t(key))
        # earth_specific_bias 是一组用于优化的参数
        EarthSpecificBias = self.earth_specific_bias[self.position_index]
        # 重塑可学习的偏置，转为和 attention 矩阵相同的 shape
        EarthSpecificBias = torch.reshape(EarthSpecificBias, shape=(self.window_size[0]*self.window_size[1]*self.window_size[2], self.window_size[0]*self.window_size[1]*self.window_size[2], self.type_of_windows, self.head_number))
        EarthSpecificBias = torch.permute(EarthSpecificBias, (2,3,0,1)) # EarthSpecificBias= (type_of_windows, head_number, win[0]win[1]win[2], win[0]win[1]win[2])
        # 目标形状为 (x.shape[0], head_number, x.shape[1], x.shape[1])
        # EarthSpecificBias = (x.shape[0]//win[0], head_number, win[0][1][2], win[0][1[2])
        # 添加 batch_size 维度，伪代码中的 [1] 表示 batch_size = 1
        EarthSpecificBias = torch.reshape(EarthSpecificBias, shape=((1,) + (EarthSpecificBias.shape)))
        # 添加偏置项
        attention = attention + EarthSpecificBias

        # 添加掩码
        # FIXME mask_attention() 没有伪代码
        attention = self.mask_attention(attention, mask)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # 计算空间混合后的 tensor
        x = torch.matmul(attention, torch.t(value))

        # 将 tensor 转为原始形状
        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, shape=original_shape)

        # 后处理线性层
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
    
class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, drop_path_ratio, heads):
        super(EarthSpecificBlock, self).__init__()
        # window size 
        self.window_size = (2, 6, 12)

        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = Mlp(dim, 0)
        self.attention = EarthAttention3D(dim, heads, 0, self.window_size)

    def forward(self, x, Z, H, W, roll):
        # 保存 skip-connection 的 shortcut
        shortcut = x
        # reshape 以计算 window attention
        x = torch.reshape(x, shape=(x.shape[0], Z, H, W, x.shape[2]))
        # FIXME 如果需要则进行零填充，需要？
        x = pad3D(X)

        # 存储形状
        ori_shape = x.shape
        
        if roll:
            # FIXME Roll x for half of the window for 3 dimensions, roll 方法需要指定 shift 和 dim 参数，此处未提供 dim
            x = torch.roll(x, shifts=[self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2])
            # FIXME gen_mask() 需自行实现
            # Generate mask of attention masks
            # If two pixels are not adjacent, then mask the attention between them
            # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
            mask = self.gen_mask(x)

        # 重构数据以计算 window attention
        x_window = torch.reshape(x, shape=(x.shape[0], Z//self.window_size[0], self.window_size[0], H//self.window_size[1], self.window_size[1], W//self.window_size[2], self.window_size[2], x.shape[-1]))

    # FIXME
    def gen_mask(x):
        return x

class EarthSpecificLayer(nn.Module):
    def __init__(self, depth, dim, drop_path_ratio_list, heads):
        '''
        网络的基本层，包括 2 或 6 个块
        '''
        super(EarthSpecificLayer,self).__init__()
        self.depth = depth
        self.blocks = []
        # 构建基本块
        for i in range(depth):
            self.blocks.append(EarthSpecificBlock(dim, drop_path_ratio_list[i], heads))
      

class PanguModel:
    def __init__(self):
        drop_path_list = torch.linspace(0, 0.2, 8)
        self._input_layer = PatchEmbedding((2, 4, 4), 192)

        self.layer1 = EarthSpecificLayer


def Train():
    '''
    Training code
    '''
    model = PanguModel()


# FIXME
def LoadConstantMask():
    return land_mask, soil_type, topography