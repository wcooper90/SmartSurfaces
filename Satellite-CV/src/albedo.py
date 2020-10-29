"""
find the brightest pixel (hopefully white), divide average brightness
by that brightest brightness, then multiply 0.65 * that. Makes the assumption
that the brightest areas in the image are comparable to brightness of office
paper.

Say if brightest point in the image is > 240, then the assumption about office
paper holds. If it's in the range 220-240, lower 0.65 to 0.6 or 0.55. Create these ranges
appropriately. Most images will have a brighest point of above 240. 
"""
