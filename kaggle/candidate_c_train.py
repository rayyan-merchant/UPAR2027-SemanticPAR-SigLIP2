"""Isolated augmentation switch; delegates unchanged training to upar.train."""
from upar.robustness import install
install()
from upar.train import main
if __name__=='__main__':main()
