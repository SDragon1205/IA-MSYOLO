# Dataset Installation
## Path of Garmin dataset
```
data_selection =>   https://drive.google.com/drive/u/0/folders/12nmGKPcq1AaZ5q_5muylp8Vd5-GHMTDm
data_selection2 =>  https://drive.google.com/drive/u/0/folders/12nmGKPcq1AaZ5q_5muylp8Vd5-GHMTDm
data_selection3 =>  https://drive.google.com/drive/u/0/folders/12nmGKPcq1AaZ5q_5muylp8Vd5-GHMTDm
night_dataset =>    https://drive.google.com/drive/u/0/folders/1lj5JwtsleQu3-_WpuwgQTk1bW1B28lPX
all the annotations => https://drive.google.com/drive/u/0/folders/1rxKecPm5zVBrXlGKzr63cuE4UDfxmNxj
```

note that data_selection ~ data_selection3 and night_dataset would have following file structure after extraction
```
data_selection\
    image_000000.jpg
    image_000001.jpg
    image_000002.jpg
    ......

data_selection_2\
    image_000000.jpg
    image_000001.jpg
    image_000002.jpg
    ......

data_selection_3\
    image_000000.jpg
    image_000001.jpg
    image_000002.jpg
    ......

night_dataset\
    2021_1130_1205_night\
        GRMN2215.MP4\
            *******.jpg
            *******.jpg
        GRMN2216.MP4\
        GRMN2217.MP4\
        .......
```
Please re-organize the file structure follow the format below
```
data_selection\
    images\
        image_000000.jpg
        image_000001.jpg
        image_000002.jpg
        ......
    anno\
        train_1cls.txt
        val_1cls.txt

data_selection_2\
    images\
        image_000000.jpg
        image_000001.jpg
        image_000002.jpg
        ......
    anno\
        train_1cls.txt
        val_1cls.txt

data_selection_3\
    images\
        image_000000.jpg
        image_000001.jpg
        image_000002.jpg
        ......
    anno\
        train_1cla.txt
        val_1cls.txt
night_dataset\
    images\
        2021_1130_1205_night\
            GRMN2215.MP4\
                *******.jpg
                *******.jpg
            GRMN2216.MP4\
            GRMN2217.MP4\
            .......
    anno\
        train_1cls.txt
        val_1cls.txt
```


## Path of TW-COCO dataset
```
Taiwan_trafficlight.v1.coco => https://drive.google.com/drive/u/0/folders/1pz02qAsdiK8m42ceZN1K1DUgpq97YDKB
# convert label script
json_to_one_txt.py => https://drive.google.com/drive/u/0/folders/1pz02qAsdiK8m42ceZN1K1DUgpq97YDKB
```

note that Taiwan_trafficlight.v1-taiwan_trafficlight.coco.zip would have following file structure after extraction
```
Taiwan_trafficlight.v1-taiwan_trafficlight.coco\
    train\
        _annotations.coco.json
        ****.jpg
        ****.jpg
        ......
    README.robotflow.txt
```
re-organize the file structure to the following format
```
Taiwan_trafficlight.v1-taiwan_trafficlight.coco\
    images\
        ****.jpg
        ****.jpg
        ......
    _annotations.coco.json
    README.robotflow.txt
    json_to_one_txt.py # download the python file from google drive
```
run json_to_one_txt.py to generate train_1cls.txt and val_1cls.txt





## Overall Dataset structure
```
yolov4-tflite
    \android
    \core
    \data
    \mAP
    \scripts
datasets
    \data_selection
        \anno
            \train_1cls.txt
            \train_3cls.txt
            \val_1cls.txt
            \val_1cls.txt
        \images
            \list of image ......
    \data_selection_2
        \anno   (follow previous format)
        \images (follow previous format)
    \data_selection_3
        \anno   (follow previous format)
        \images (follow previous format)
    \data_selection_mix
        \anno   (follow previous format)
    \Taiwan_trafficlight.v1.coco
        \anno   (follow previous format)
        \images (follow previous format)
    \night_dataset
        \anno   (follow previous format)
        \images (follow previous format)
```