# Kid can label
Let your kids label data for you

## Pre-requirements
* Docker Engine
* NVIDIA Docker Container

## Download Model
| Model | FILE | Size 
| --- | --- | ---
| [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | sam_vit_h_4b8939.pth | 2.4G
| [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | sam_vit_l_0b3195.pth | 1.2G
| [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | sam_vit_b_01ec64.pth | 358M

* Download with WGET
  ```bash
  NAME="sam_vit_b_01ec64.pth"
  wget -P ./models \
  https://dl.fbaipublicfiles.com/segment_anything/${NAME}
  ```

## Build 
```bash
docker compose build kidcanlabel
```

## Usage
```bash
xhost + > /dev/null
docker compose run kidcanlabel
python3 kid_can_label.py -i <input>
```
