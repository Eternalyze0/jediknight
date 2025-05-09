import torch

class FuturePredictor(torch.nn.Module): # predicts the future given the present state and action
	pass
class ActionPredictor(torch.nn.Module): # predicts the action given the present and future states
	pass
class StateEmbedder(torch.nn.Module): # embeds the state (image & snapshot)
	pass
class ImageEmbedder(torch.nn.Module): # embeds the image (raw game pixels)
	pass
class SnapshotEmbedder(torch.nn.Module): # embeds the snapshot (core game stats)
	pass
class Actor(torch.nn.Module): # chooses actions freely via its own curiosity
	pass
