
output_dir=/home/chunyl/project/OUTPUT_DIR/GPT3 # the path that the generated gpt3 knowledge is saved
apikey=XXXX # Please use your GPT3 API key

ds='cifar10'
# ['eurosat-clip','country211','kitti-distance','oxford-iiit-pets','ping-attack-on-titan-plus','ping-whiskey-plus','rendered-sst2','resisc45-clip','voc2007classification','caltech101','cifar10','cifar100','dtd','fer2013','fgvc-aircraft-2013b','flower102','food101','gtsrb','hateful-memes','mnist','patchcamelyon','stanfordcar']


cd vision_benchmark


python commands/extract_gpt3_knowledge.py --ds resources/datasets/$ds.yaml --apikey $apikey --n_shot 3 --n_ensemble 5 \
--target local DATASET.ROOT $output_dir/datasets/ds OUTPUT_DIR $output_dir/log 



# pip install openai
# pip install nltk, spacy
# python -m spacy download en