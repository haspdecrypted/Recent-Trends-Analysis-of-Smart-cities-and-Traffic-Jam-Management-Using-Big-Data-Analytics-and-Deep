from imageai.Detection import VideoObjectDetection
import os
import json
execution_path = os.getcwd()
logImageLink = open("perSecondLog.txt", "w")
def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("Output average count for unique objects in the last second: ", average_output_count)
    logImageLink.write(json.dumps(average_output_count)+'/n')
    


video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
video_detector.loadModel()
custom = video_detector.CustomObjects(bicycle=True,   car=True,   motorcycle=True, bus=True, truck=True)

video_detector.detectObjectsFromVideo(custom_objects=custom, input_file_path=os.path.join(execution_path, "Road traffic video for object recognition.mp4"), output_file_path=os.path.join(execution_path, "video_second_analysis") ,  frames_per_second=1, per_second_function=forSeconds, log_progress=True,  minimum_percentage_probability=50)
logImageLink.close()
