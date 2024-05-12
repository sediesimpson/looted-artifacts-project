import argparse

parser = argparse.ArgumentParser(description = 'Running Baseline Models')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--dataset', type=str, default="Test Dataset 2")

args = parser.parse_args()

# if __name__ == '__main__':
# main()
