


from machinelearning.attmap.utils import *
from systemtools.basics import *
from systemtools.file import *
from systemtools.location import *
from systemtools.printer import *
import copy
from machinelearning.utils import *


class COLOR:
	# red = "255, 50, 50"
	# red = "242, 82, 82" # hjred
	red = "255, 100, 100" # hjred
	# green = "50, 255, 50"
	# green = "0, 155, 149" # hjgreen
	green = "50, 255, 149"
	# blue = "50, 50, 255"
	blue = "0, 122, 204" # hjblue

def buildToken(id, token, strengths=[1.0], colors=[COLOR.red, COLOR.green, COLOR.blue], minAlpha=0.1, maxAlpha=0.8):
	if not isinstance(strengths, list):
		strengths = [strengths]
	if not isinstance(colors, list):
		colors = [colors]
	colors = colors[:len(strengths)]
	id = "e" + str(id)
	el = ""
	el += '<span class="token" id="' + id + '">'
	el += '<span class="text">' + token + '</span>' + "\n"
	for i in range(len(strengths)):
		el += '<span class="att' + str(i) + '"></span>' + "\n"
	el += '</span>'

	strengths = copy.deepcopy(strengths)
	for i in range(len(strengths)):
		strengths[i] = truncateFloat(linearScore(strengths[i], x1=0.0, x2=1.0, y1=minAlpha, y2=maxAlpha, stayBetween0And1=True), 2)

	if len(strengths) == 1:
		tops = [0]
		heights = [100]
	elif len(strengths) == 2:
		tops = [10, 50]
		heights = [50, 50]
	elif len(strengths) == 3:
		tops = [10, 33.3 + 5, 66.6]
		heights = [33.3, 33.3, 33.3]

	style = ""
	for i in range(len(tops)):
		currentStyle = ""
		currentStyle += "#" + id + " > .att" + str(i) + " {\n"
		currentStyle += "position: absolute;" + "\n"
		currentStyle += "top: " + str(tops[i]) + "%;" + "\n"
		currentStyle += "height: " + str(heights[i]) + "%;" + "\n"
		currentStyle += "left: -2px;" + "\n"
		currentStyle += "width: 100%;" + "\n"
		currentStyle += "background: linear-gradient(0.5turn, rgba(0, 255, 0, 0.0), rgba(" + colors[i] + ", " + str(strengths[i]) + "), rgba(0, 255, 0, 0.0));" + "\n"
		currentStyle += "}\n"
		style += currentStyle
	return (el, style)





def generateAttentionMap(sentences, attentions, *args, path=None, sentencesAmount=None, maxLetters=None, masks=[None, MASK_TOKEN], tmpDirName="attmap", **kwargs):
	if sentencesAmount is not None or maxLetters is not None:
		raise Exception("Not yet implemented")
	if not isinstance(masks, list):
		masks = [masks]
	if sentences is None or len(sentences) == 0:
		return None
	if isinstance(sentences[0], list):
		sentences = flattenLists(sentences)
	if not isinstance(attentions[0], list):
		attentions = [attentions]
	# We check the length of both sentences and attentions:
	assert len(attentions[0]) == len(sentences)
	# FORMULE : (somme des (proba au carré)) / nb de mots
	# plutot faire maxLetters au lieu de sentencesAmount... et se debrouiller pour partager la place à tous les attentions : juste piocher dans le top de chaque attention un par un et on s'arrete quand on a depassé maxLetters
	# We remove masks:
	newSentences = []
	newAttentions = []
	for u in range(len(attentions)):
		newAttentions.append([])
	for i in range(len(sentences)):
		if sentences[i] not in masks:
			newSentences.append(sentences[i])
			for u in range(len(attentions)):
				newAttentions[u].append(attentions[u][i])
	sentences = newSentences
	attentions = newAttentions
	# We normalize attention scores:
	for i in range(len(attentions)):
		attentions[i] = minMaxNormalize(attentions[i])
	# Then we build the html:
	page = buildAttentionHtml(sentences, attentions, *args, **kwargs)
	randomId = getRandomStr()
	htmlPath = tmpDir(tmpDirName) + "/" + randomId + ".html"
	strToFile(page, htmlPath)
	if path is None:
		path = tmpDir(tmpDirName) + "/" + randomId + ".png"
	pngPath = html2png(htmlPath, destPath=path)
	remove(htmlPath)
	pngPath = cropPNG(pngPath)
	return pngPath

def showAttentionMap(*args, **kwargs):
	path = generateAttentionMap(*args, **kwargs)
	from IPython.display import Image, display
	img = Image(filename=path) 
	display(img)

def buildAttentionHtml(tokens, attentions, *args, **kwargs):
	styles = []
	elements = []
	for i in range(len(tokens)):
		token = tokens[i]
		attention = [current[i] for current in attentions]
		(el, style) = buildToken(i, token, attention)
		styles.append(style)
		elements.append(el)
	styles = "\n".join(styles)
	elements = "<div>" + "\n".join(elements) + "</div>"
	page = buildPage(styles, elements, *args, **kwargs)
	return page



if __name__ == '__main__':
	
	tokens = [None, None, None, "the", "thing", "you", "can", "not", "understand", "."] * 10
	attentions = [1.0, 1.0, 1.0, 0.0, 0.5, 0.1, 0.9, 0.2, 1.0, 0.0] * 10
	# attentions = \
	# [
	# 	[0.0, 0.5, 0.1, 0.9, 0.2, 1.0, 0.0] * 10,
	# 	[0.4, 0.2, 0.1, 0.4, 0.8, 0.3, 0.5] * 10
	# ]
	attentions = \
	[
		[1.0, 1.0, 1.0, 0.0, 0.5, 0.1, 0.9, 0.2, 1.0, 0.0] * 10,
		[1.0, 1.0, 1.0, 0.4, 0.2, 0.1, 0.4, 0.8, 0.9, 0.5] * 10,
		[1.0, 1.0, 1.0, 0.2, 0.9, 0.5, 0.2, 0.6, 0.3, 1.0] * 10
	]

	print(generateAttentionMap(tokens, attentions, path=tmpDir() + "/test.png"))

	# bp((styles, elements))


