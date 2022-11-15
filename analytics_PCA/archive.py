import numpy as np
import pandas as pd


class Archive:
    # Make Archive
    def __init__(self):
        print("### Archive TXT ###")
        self.archive_var = []
        self.archive_obj = []
    
    def _add(self, var, obj):
        if len(self.archive_var) == 0:
            self.archive_var = np.array([var.copy()])
            self.archive_obj = np.array([obj])
        else:
            if np.any(np.all(self.archive_var == var, axis=1)): # 重複判定
                return False
            # if distance.cdist(self.archive_var, [var]).min() < 1e-5:
            #     return False
            self.archive_var = np.vstack((self.archive_var, var))
            self.archive_obj = np.hstack((self.archive_obj, obj))
        return True
    
    def _stack(self):
        self.archive_stack = np.vstack((self.archive_var.T, self.archive_obj.T))

    def _sort(self):
        idx = np.argsort(self.archive_obj)
        self.archive_obj = self.archive_obj[idx]
        self.archive_var = self.archive_var[idx]

    def getVarBestFit(self,m):
        if m <= self.getArchiveSize():
            return self.archive_var
        idx = np.argpartition(self.archive_obj,m-1)[:m]
        return self.archive_var[idx]

    def getObjBestFit(self,m):
        if m <= self.getArchiveSize():
            return self.archive_obj
        idx = np.argpartition(self.archive_obj,m-1)[:m]
        return self.archive_obj[idx]

    def getArchiveSize(self):
        return len(self.archive_obj)
    
    def _log_archive(self,eval):
        df_archive = []
        for i in range(self.getArchiveSize()):
            log_list = {"eval":eval,"obj":self.archive_obj[i]}
            for dim in range(len(self.archive_var[i])):
                log_list["x{}".format(dim)] = self.archive_var[i,dim]
            df_archive.append(log_list)
        return df_archive