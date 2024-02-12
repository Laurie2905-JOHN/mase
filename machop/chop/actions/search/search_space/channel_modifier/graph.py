# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
from torch import nn
from ..base import SearchSpaceBase
from .....passes.graph.transforms.channel_modifier import (
    CHANNEL_OP,
    redefine_transform_pass,
)

from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target, get_parent_name
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict
import torch


# Default configuration for channel modification, specifying a fallback
# channel_multiplier and an optional operation name.
DEFAULT_CHANNEL_MODIFIER_CONFIG = {
    "config": {
        "name": None,
        "channel_multiplier": 1,
    }
}

class ChannelSizeModifier(SearchSpaceBase):
    """
    A class to define pre-training search space for modifying the channel sizes within a model's graph.
    This is particularly useful for exploring different model architectures dynamically.
    """
    
    def _post_init_setup(self):
        """
        Setup method to initialize class variables and move the model to the GPU to avoid tensor on same device error.
        """
        self.model.to("cuda")  # Move model to CUDA
        self.mg = None  # Placeholder for the modified graph, to be generated
        self._node_info = None  # Placeholder for node information, to be populated
        self.default_config = DEFAULT_CHANNEL_MODIFIER_CONFIG  # Set the default channel modifier configuration

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        """
        Rebuilds the model according to the sampled configuration, with an option to set the model
        in evaluation mode or training mode.

        :param sampled_config: Configuration dict for modifying channel sizes.
        :param is_eval_mode: Flag to set the model in evaluation mode; defaults to True.
        :return: Modified graph with new channel sizes.
        """
        if is_eval_mode:
            self.model.eval()  # Set model to evaluation mode
        else:
            self.model.train()  # Set model to training mode

        # Self did not work, neither deepcopy so this was a workaround.
        mg = MaseGraph(self.model)
        mg, _ = init_metadata_analysis_pass(mg, None)
        mg, _ = add_common_metadata_analysis_pass(
            mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
        )
        
        if sampled_config is not None:
            mg, _ = redefine_transform_pass(mg, {"config": sampled_config})  # Apply channel size modifications
        mg.model.to(self.accelerator)  # Move the modified graph's model to the specified device
        return mg

    def build_search_space(self):
        """
        Constructs the search space for channel size modification by mapping node names to their
        corresponding types and operations, and determining the choices of channel multipliers.

        :return: None. The method updates the instance's choice mappings.
        """

        # Build a mapping from node name to mase_type and mase_op.
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)

        # Build the search space
        choices = {}
        seed = self.config["seed"]

        for node in mase_graph.fx_graph.nodes:
            if node.name in seed:
                choices[node.name] = deepcopy(seed[node.name])
            else:
                choices[node.name] = deepcopy(seed["default"])

        # Flatten the choices dictionary for easier access and manipulation
        flatten_dict(choices, flattened=self.choices_flattened)
        # Calculate the length of choices for each node and store it
        self.choice_lengths_flattened = {k: len(v) for k, v in self.choices_flattened.items()}
        
    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["setup"]["by"]
        return config

