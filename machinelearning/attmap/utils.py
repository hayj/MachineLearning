def buildPage(style, content, width=690, fontSize=12, lineHeight=0.8):
	page = "<!DOCTYPE html><html><head><title>Title of the document</title><style>"
	page +=\
	"""
		*
		{
			margin: 0;
			padding: 0;
		}

		body
		{
			width: """ + str(width) + """px;
			margin-left: 10px;
			margin-top: 10px;
		}

		body > div
		{
			//text-align: justify;
			line-height: """ + str(lineHeight) + """;
		}

		.token
		{
			position: relative;
			font-size: """ + str(fontSize) + """px;
			color: rgba(20, 20, 20, 1.0);
			z-index: 3;
		}
	"""
	page += style
	page += "</style></head><body>"
	page += content
	page += "</body></html>"
	return page

