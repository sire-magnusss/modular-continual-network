"""
Standard continual learning metrics.

After training on T tasks, we have an accuracy matrix R where:
  R[i][j] = accuracy on task j after training on task i

From this matrix we compute:
  - Average Accuracy (AA)     : mean accuracy on all seen tasks after final task
  - Backward Transfer (BWT)   : how much learning new tasks hurt old ones (negative = forgetting)
  - Forgetting Measure (FM)   : average max-accuracy drop per task
"""

import numpy as np


class MetricTracker:
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # R[i][j] = accuracy on task j after training task i
        self.R = np.full((num_tasks, num_tasks), np.nan)

    def update(self, trained_up_to: int, task_id: int, accuracy: float):
        """Record accuracy on task_id after having trained up to trained_up_to."""
        self.R[trained_up_to][task_id] = accuracy

    def average_accuracy(self) -> float:
        """Mean accuracy across all tasks after training on the final task."""
        last_row = self.R[self.num_tasks - 1]
        valid = last_row[~np.isnan(last_row)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def backward_transfer(self) -> float:
        """
        BWT = (1 / T-1) * sum_{i=1}^{T-1} [ R[T-1][i] - R[i][i] ]
        Negative BWT means forgetting. Positive means old tasks improved (rare).
        """
        if self.num_tasks < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(self.num_tasks - 1):
            if not np.isnan(self.R[self.num_tasks - 1][i]) and not np.isnan(self.R[i][i]):
                total += self.R[self.num_tasks - 1][i] - self.R[i][i]
                count += 1
        return float(total / count) if count > 0 else 0.0

    def forgetting_measure(self) -> float:
        """
        FM = (1 / T-1) * sum_{i=1}^{T-1} [ max_{j<=T-1} R[j][i] - R[T-1][i] ]
        How much accuracy dropped from peak for each task.
        """
        if self.num_tasks < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(self.num_tasks - 1):
            col = self.R[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) >= 2:
                peak = np.max(valid)
                final = self.R[self.num_tasks - 1][i]
                if not np.isnan(final):
                    total += peak - final
                    count += 1
        return float(total / count) if count > 0 else 0.0

    def summary(self) -> dict:
        return {
            "average_accuracy": self.average_accuracy(),
            "backward_transfer": self.backward_transfer(),
            "forgetting_measure": self.forgetting_measure(),
            "accuracy_matrix": self.R.tolist(),
        }

    def print_matrix(self):
        print("\n--- Accuracy Matrix R[trained_up_to][task] ---")
        header = "      " + "  ".join(f"T{j}" for j in range(self.num_tasks))
        print(header)
        for i in range(self.num_tasks):
            row_str = f"After T{i}: "
            for j in range(self.num_tasks):
                val = self.R[i][j]
                row_str += f"{val*100:5.1f}  " if not np.isnan(val) else "  ---  "
            print(row_str)
        print(f"\nAverage Accuracy : {self.average_accuracy()*100:.2f}%")
        print(f"Backward Transfer: {self.backward_transfer()*100:.2f}%")
        print(f"Forgetting Measure: {self.forgetting_measure()*100:.2f}%")
