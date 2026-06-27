# Trained Model

The trained ResNet50 model is **not included** in this repository because its size exceeds GitHub's file size limit (100 MB).

## Download Model

Download the trained model from the following Google Drive link:

**Google Drive Link**

https://drive.google.com/file/d/1CJSsHiNOnXQl-kmLfROGHfxXXBe6jUD3/view?usp=drive_link

After downloading, place the model file inside this directory.

```text
models/
└── resnet50_finetuned_model.keras
```

The application expects the model at:

```text
models/resnet50_finetuned_model.keras
```

Once the model is placed in this directory, the application can be started normally using:

```bash
streamlit run app.py
```
