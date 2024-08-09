CUDA_VISIBLE_DEVICES=0 python ptvid/src/bert/main.py --domain journalistic
CUDA_VISIBLE_DEVICES=0 python ptvid/src/bert/main.py --domain literature
CUDA_VISIBLE_DEVICES=1 python ptvid/src/bert/main.py --domain legal
CUDA_VISIBLE_DEVICES=2 python ptvid/src/bert/main.py --domain politics
CUDA_VISIBLE_DEVICES=3 python ptvid/src/bert/main.py --domain web
CUDA_VISIBLE_DEVICES=0 python ptvid/src/bert/main.py --domain social_media
