from ultralyticsplus import YOLO,render_result

model=YOLO('keremberke/yolov8m-pothole-segmentation')

model.overrides['conf']=0.25
model.overrides['iou']=0.45
model.overrides['agnostic_nms']=False
model.overrides['max_det']=1000

image='https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
result=model.predict(image)

print(result[0].boxes)
print(result[0].masks)

render=render_result(model=model,image=image,result=result[0])
render.show()

