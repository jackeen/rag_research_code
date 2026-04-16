"""
The tester of arg parser.
"""

import argparse




if __name__ == '__main__':
    p = argparse.ArgumentParser(
        prog='test_arg',
        description='Test the arguments parser',
    )

    p.add_argument('order', type=str, help='order')

    arg = p.parse_args()
    print(arg.order)
