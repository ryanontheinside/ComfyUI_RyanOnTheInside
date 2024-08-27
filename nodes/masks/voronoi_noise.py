import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#NOTE credit for this class and much of the heavy lifting in the Flex Voronoi nodes goes to Alan Huang 
# https://github.com/alanhuang67/
class VoronoiNoise(nn.Module):
    def __init__(self, width, height, scale, detail, seed, randomness, X=[0], Y=[0], distance_metric='euclidean', batch_size=1, device='cpu'):
        super(VoronoiNoise, self).__init__()
        self.width, self.height = width, height
        self.scale = torch.tensor(self._adjust_list_length(scale, batch_size), dtype=torch.float, device=device)
        self.detail = self._adjust_list_length(detail, batch_size)
        self.seed = self._adjust_list_length(seed, batch_size)
        self.randomness = torch.tensor(self._adjust_list_length(randomness, batch_size), dtype=torch.float, device=device)
        self.X = torch.tensor(self._adjust_list_length(X, batch_size), dtype=torch.float, device=device)
        self.Y = torch.tensor(self._adjust_list_length(Y, batch_size), dtype=torch.float, device=device)
        self.distance_metric = distance_metric
        self.batch_size = batch_size
        self.device = device

    @staticmethod
    def _adjust_list_length(lst, length):
        return lst + [lst[-1]] * (length - len(lst)) if len(lst) < length else lst

    def forward(self):
        noise_batch = []
        for b in range(self.batch_size):
            torch.manual_seed(self.seed[b])
            center_x = self.width // 2
            center_y = self.height // 2
            sqrt_detail = int(np.sqrt(self.detail[b]))
            spacing = max(self.width, self.height) / sqrt_detail
            offsets_x = torch.arange(-sqrt_detail // 2, sqrt_detail // 2 + 1, device=self.device) * spacing
            offsets_y = torch.arange(-sqrt_detail // 2, sqrt_detail // 2 + 1, device=self.device) * spacing
            grid_x, grid_y = torch.meshgrid(offsets_x, offsets_y, indexing='xy')
            points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            random_offsets = (torch.rand_like(points) * 2 - 1) * self.randomness[b] * spacing / 2
            points += random_offsets
            points[len(points) // 2] = torch.tensor([0, 0], device=self.device)
            points *= self.scale[b]
            points += torch.tensor([self.X[b], self.Y[b]], device=self.device)
            points += torch.tensor([center_x, center_y], device=self.device)
            x_coords = torch.arange(self.width, device=self.device)
            y_coords = torch.arange(self.height, device=self.device)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            grid = torch.stack([grid_x, grid_y], dim=-1).float()

            if self.distance_metric == 'euclidean':
                distances = torch.sqrt(((grid.unsqueeze(2) - points) ** 2).sum(dim=-1))
            elif self.distance_metric == 'manhattan':
                distances = torch.abs(grid.unsqueeze(2) - points).sum(dim=-1)
            elif self.distance_metric == 'chebyshev':
                distances = torch.abs(grid.unsqueeze(2) - points).max(dim=-1).values
            elif self.distance_metric == 'minkowski':
                p = 3
                distances = (torch.abs(grid.unsqueeze(2) - points) ** p).sum(dim=-1) ** (1/p)
            elif self.distance_metric == 'elliptical':
                # 确定长轴和短轴
                if self.width > self.height:
                    a = self.width / self.height
                    b = 1
                else:
                    a = 1
                    b = self.height / self.width
                # 放大或缩小椭圆的比例，使效果更显著
                scale_factor_a = 3.5  # 调整长轴比例以增强效果
                scale_factor_b = 1  # 调整短轴比例以增强效果
                a *= scale_factor_a
                b *= scale_factor_b
                distances = torch.sqrt(((grid.unsqueeze(2) - points) ** 2 / torch.tensor([a, b], device=self.device)).sum(dim=-1))

            elif self.distance_metric == 'kaleidoscope_star':
                def kaleidoscope_star_shape(theta):
                    return torch.abs(torch.sin(8 * theta))  # 调整8以生成不同数量的星点
                
                theta = torch.atan2(grid[..., 1].unsqueeze(2) - points[..., 1], grid[..., 0].unsqueeze(2) - points[..., 0])
                radius = torch.sqrt((grid[..., 0].unsqueeze(2) - points[..., 0]) ** 2 + (grid[..., 1].unsqueeze(2) - points[..., 1]) ** 2)
                star_radius = kaleidoscope_star_shape(theta)
                distances = torch.abs(radius - star_radius * radius / star_radius.max())
                distances = 1 - distances / distances.max()  # Adjust contrast
            elif self.distance_metric == 'kaleidoscope_wave':
                def kaleidoscope_wave_shape(theta):
                    return 1 + 0.3 * torch.sin(4 * theta + radius / 10)  # 结合角度和半径生成波浪
                
                theta = torch.atan2(grid[..., 1].unsqueeze(2) - points[..., 1], grid[..., 0].unsqueeze(2) - points[..., 0])
                radius = torch.sqrt((grid[..., 0].unsqueeze(2) - points[..., 0]) ** 2 + (grid[..., 1].unsqueeze(2) - points[..., 1]) ** 2)
                wave_radius = kaleidoscope_wave_shape(theta)
                distances = torch.abs(radius - wave_radius * radius / wave_radius.max())
                distances = 1 - distances / distances.max()  # Adjust contrast
            elif self.distance_metric == 'kaleidoscope_radiation_α':
                theta = torch.atan2(grid[..., 1].unsqueeze(2) - points[..., 1], grid[..., 0].unsqueeze(2) - points[..., 0])
                radius = torch.sqrt((grid[..., 0].unsqueeze(2) - points[..., 0])**2 + (grid[..., 1].unsqueeze(2) - points[..., 1])**2)
                distances = torch.abs(torch.sin(6 * theta) * radius)
                distances = 1 - distances / distances.max()  # 调整对比度
            elif self.distance_metric == 'kaleidoscope_radiation_β':
                def kaleidoscope_2_shape(theta):
                    return 1 + 0.5 * torch.sin(5 * theta)
                theta = torch.atan2(grid[..., 1].unsqueeze(2) - points[..., 1], grid[..., 0].unsqueeze(2) - points[..., 0])
                radius = torch.sqrt((grid[..., 0].unsqueeze(2) - points[..., 0]) ** 2 + (grid[..., 1].unsqueeze(2) - points[..., 1]) ** 2)
                kaleidoscope_2_radius = kaleidoscope_2_shape(theta)
                distances = torch.abs(radius - kaleidoscope_2_radius * radius / kaleidoscope_2_radius.max())
                distances = 1 - distances / distances.max()  # Adjust contrast
            elif self.distance_metric == 'kaleidoscope_radiation_γ':
                def kaleidoscope_diamond_shape(theta):
                    # 使用菱形对称结构生成形状
                    return torch.abs(torch.sin(4 * theta))  # 调整4以生成不同数量的菱角
                
                theta = torch.atan2(grid[..., 1].unsqueeze(2) - points[..., 1], grid[..., 0].unsqueeze(2) - points[..., 0])
                radius = torch.sqrt((grid[..., 0].unsqueeze(2) - points[..., 0]) ** 2 + (grid[..., 1].unsqueeze(2) - points[..., 1]) ** 2)
                diamond_radius = kaleidoscope_diamond_shape(theta)
                distances = torch.abs(radius - diamond_radius * radius / diamond_radius.max())
                distances = 1 - distances / distances.max()  # 调整对比度
                
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

            min_distances, _ = distances.min(dim=-1)
            single_noise = min_distances
            single_noise_flat = single_noise.view(-1)
            local_min = single_noise_flat.min()
            local_max = single_noise_flat.max()
            normalized_noise = (single_noise - local_min) / (local_max - local_min)
            final_output = normalized_noise.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).cpu()
            noise_batch.append(final_output)

        return torch.cat(noise_batch, dim=0)
