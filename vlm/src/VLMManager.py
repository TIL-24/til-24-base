from typing import List
from io import BytesIO
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class VLMManager:
    def __init__(self):
        # initialize the model here
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
        pass

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model

        processed_image = Image.open(BytesIO(image))
        prompt = (f"Output the bounding box of {caption} in the image "
                  f"using the [x_min, y_min, x_max, y_max] format. "
                  f"Your reply should only contain the bounding box expression. ")
        full_prompt = "[INST] <image>\n" + prompt + "[/INST]"

        inputs = self.processor(full_prompt, processed_image, return_tensors="pt")
        output = self.model.generate(**inputs)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        print(response)

        return [0, 0, 0, 0]


if __name__ == "__main__":
    manager = VLMManager()
    with open("image.jpg", "rb") as f:
        image_bytes = f.read()
        print(manager.identify(image_bytes, "iron man"))
