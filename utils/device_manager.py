import torch

import torch


class DeviceManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def move_model_to_device(self, model):
        """
        Move a model to the selected device.

        Args:
            model (torch.nn.Module): The PyTorch model to be moved.

        Returns:
            torch.nn.Module: The model after being moved to the device.
        """
        return model.to(self.device)

    def move_data_to_device(self, data):
        """
        Recursively move data (tensor, list, tuple, dict) to the selected device and ensure it is of type float32.

        Args:
            data (Tensor, list, tuple, dict): Data to be moved to the device.

        Returns:
            Same type as input: Data after being moved to the device and converted to float32 if necessary.
        """
        if isinstance(data, (list, tuple)):
            return [self.move_data_to_device(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.move_data_to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            # Ensure tensor is float32 and move it to the device
            if data.dtype != torch.float32:
                data = data.float()
            return data.to(self.device, non_blocking=True)
        else:
            raise TypeError(f"Unsupported data type {type(data)} for moving to device.")


# 实例化 DeviceManager
DM = DeviceManager()

# 使用示范：
# from device_manager import DM
# model = DM.move_model_to_device(MyModel())
# input_tensor = DM.move_data_to_device(torch.randn(64, 10))
# output = model(input_tensor)
