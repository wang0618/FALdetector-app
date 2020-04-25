# [FALdetector app](https://github.com/wang0618/FALdetector-app)

基于 [FALdetector](https://github.com/PeterWang512/FALdetector/) 项目构建的Web应用，支持在线监测图片是否经过PS并尝试恢复PS前的原始图片。

## Demo

[![Try in PWD](https://cdn.rawgit.com/play-with-docker/stacks/cff22438/assets/images/button.png)](http://play-with-docker.com?stack=https://raw.githubusercontent.com/wang0618/FALdetector-app/master/docker-compose.yml) 

使用 play-with-docker 运行demo容器app，需要等待几秒钟部署完成后再去访问80端口


## 本地运行

### 通过Docker运行

```docker run --name faldetector -p 80:80 wangweimin/faldetector-app```

### 通过代码运行
**Clone project**
```bash
git clone --recursive https://github.com/wang0618/FALdetector-app.git
```

**Install packages**
 - Install PyTorch (pytorch.org)
 - `pip install -r FALdetector/requirements.txt`
 - `pip install -r requirements.txt`

**Download model weights**
Run `bash FALdetector/weights/download_weights.sh`

**Start app**
Run `python start_app.py`