
"""
Following https://www.pythonforbeginners.com/argparse/argparse-tutorial
"""

# 1
# import argparse
# parser = argparse.ArgumentParser()
# parser.parse_args()


# 2
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("echo") 	# naming it "echo"
# args = parser.parse_args()	# returns data from the options specified (echo)
# print(args.echo)


# # 3
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("echo", help="echo the string you use here")
# args = parser.parse_args()
# print(args.echo)


# # 4
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("square", help="display a square of a given number",
#                     type=int)
# args = parser.parse_args()
# print(args.square**2)


# 5
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--verbose", help="increase output verbosity",
#                     action="store_true")
# args = parser.parse_args()
# if args.verbose:
#     print("verbosity turned on")


# 6
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("x", type=int, help="the base")
# parser.add_argument("y", type=int, help="the exponent")
# parser.add_argument("-v", "--verbosity", action="count", default=0)
# args = parser.parse_args()
# answer = args.x**args.y
# if args.verbosity >= 2:
#     print("{} to the power {} equals {}".format(args.x, args.y, answer))
# elif args.verbosity >= 1:
#     print("{}^{} == {}".format(args.x, args.y, answer))
# else:
#     print(answer)


# mine1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--square', help="number to be squared", default=1, type=int)
parser.add_argument('-m1', '--mult1', help="first mult", default=1, type=int)
parser.add_argument('-m2', '--mult2', help="second mult", default=1, type=int)
args = parser.parse_args()
print(args.square**2)
print(args.mult1*args.mult2)