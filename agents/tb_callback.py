
# agents/tb_callback.py
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class TBMetricsCallback(BaseCallback):
    """
    Logs custom environment metrics (coverage_fraction, avg_wait) to TensorBoard.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.writer = None

    def _on_training_start(self) -> None:
        # Initialize TensorBoard writer
        log_dir = self.logger.dir
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        # env is a VecEnv
        for i, info in enumerate(self.locals["infos"]):
            coverage = info.get("coverage_fraction")
            avg_wait = info.get("avg_wait")
            if coverage is not None:
                self.writer.add_scalar("env/coverage_fraction", coverage, self.num_timesteps)
            if avg_wait is not None:
                self.writer.add_scalar("env/avg_wait", avg_wait, self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()
