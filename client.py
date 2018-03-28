import requests
import json

if __name__ == '__main__':
	addr = 'http://localhost:5000'
	test_url = addr + '/describe_image'

	# prepare headers for http request
	content_type = 'image/jpeg'
	headers = {'content-type': content_type}

	with open('sample_images/COCO_val2014_000000224477.jpg','rb') as f:
		img_data=f.read()
	# send http request with image and receive response
	response = requests.post(test_url, data=img_data, headers=headers)
	response_data=json.loads(response.text)
	print(response_data)