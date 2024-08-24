class FeaturePipe:
    def __init__(self, frame_rate, video_frames):
        self.frame_rate = frame_rate
        self.frame_count = video_frames.shape[0]
        self.height = video_frames.shape[1]
        self.width = video_frames.shape[2]

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self