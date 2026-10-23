"""Keep completed task paths fixed, including non-parameter model state."""


class FrozenTaskLifecycle:
    """Mixin for models exposing base modules and task-specific components.

    ``requires_grad=False`` alone does not freeze BatchNorm statistics or
    disable dropout. Reapply eval mode whenever a trainer calls model.train().
    """

    def train(self, mode=True):
        super().train(mode)
        if self._base_frozen:
            for module in self._base_modules_to_freeze():
                module.eval()
        for task_id in range(self._num_trained):
            for module in self._task_components(task_id):
                module.eval()
        return self

    def freeze_task(self, task_id):
        for module in self._task_components(task_id):
            module.requires_grad_(False)
            module.eval()

    def complete_task(self, task_id):
        if task_id != self._num_trained:
            raise ValueError("Tasks must be completed once, in sequential order.")
        if task_id == 0:
            self.freeze_base_encoder()
        self.freeze_task(task_id)
        self._num_trained = task_id + 1
        self.train(self.training)

    def get_extra_state(self):
        # Persist Python lifecycle flags alongside the tensor state_dict.
        return {"base_frozen": self._base_frozen,
                "num_trained": self._num_trained}

    def set_extra_state(self, state):
        self._base_frozen = False
        self._num_trained = 0
        self.requires_grad_(True)
        if state["base_frozen"]:
            self.freeze_base_encoder()
        for task_id in range(state["num_trained"]):
            self.freeze_task(task_id)
        self._num_trained = state["num_trained"]
        self.train(self.training)
