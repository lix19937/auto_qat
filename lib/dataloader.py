
import numpy as np
import torch 

class DataBase(torch.utils.data.Dataset):
    def __init__(self, input_dims: list, train_files : str = None, test_files : str = None, is_train : bool = True) -> None:
        super().__init__()
        self.in_dims = input_dims
        if train_files is None and test_files is None:
          train_filelist = [None] * 32
          test_filelist = [None] * 32
          self.data = train_filelist if is_train else test_filelist


    def __getitem__(self, index : int):
        # Reads labels
        label = np.zeros(10, dtype=np.float32)

        list_in = []
        # Reads data
        for it in self.in_dims:
          frame = np.random.rand(it[1], it[2], it[3]) 
          list_in.append(frame)
        return torch.from_numpy(list_in[0]), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

  
def get_dataloader(dataset : DataBase, batch_size : int, num_workers : int):
  loader = torch.utils.data.DataLoader(
    dataset=dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    pin_memory=True, 
    shuffle=False, 
    drop_last=False)

  base_iter = (loader.dataset.__len__() + loader.batch_size -1) // loader.batch_size
  return loader, base_iter
