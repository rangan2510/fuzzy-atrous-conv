# %% [code]
#%%
import time, datetime
import humanize
# %%
class HowLong():
    def __init__(self) -> None:
        self.now = time.time()
        self.start_time = time.time()
        self.last = time.time()
        
    def since_start(self, simple=True):
        now = time.time()
        diff = now - self.start_time
        if simple:
            print(humanize.naturaldelta(datetime.timedelta(seconds = diff)))
        else:
            print(datetime.timedelta(seconds = diff))
        self.last = time.time()

    def since_last(self, simple=True):
        now = time.time()
        diff = now - self.last
        if simple:
            print(humanize.naturaldelta(datetime.timedelta(seconds = diff)))
        else:
            print(datetime.timedelta(seconds = diff))
        self.last = time.time()


