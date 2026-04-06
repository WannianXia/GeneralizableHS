from Algo.arguments import parser
from Algo.dmc import train
import os


if __name__ == "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
	flags = parser.parse_args()
	train(flags)
