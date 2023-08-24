# RawByteClf

unzip raw.7z


## Notes



## TODO

- instead of saving both microsoft train and microsoft test for mlm, adjust the scripts to combine the clf and mlm datasets when doing mlm
- pretrain mlm on all 10000 byte chunks?
- pad vocabulary to a multiple of 8
- save logfile in the correct output directory
- adjust preprocessing script to elegantly handle OOM issues
- add 12-bit byte capabaility
  - decompose 8-bit bytes into bits; then reform into bytes
  - perform entirely in memory

