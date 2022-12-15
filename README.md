# Spam Radar

### Installation
In order to use this program you need to use our modules. In order to download all these type in your terminal with Python 3.10 installed.

`$ pip install -r requirements.txt`
This will download all the modules, this will take around a minute to 20 minutes.

### File System
In spam radar we got multiple folders and multiple files in different languages, all severves a purpose, either if it's for testing, or just holding data.

###### Files:
- main.py (Run this script in order to analyze your emails.)
- index.html followed up we also have script.js and style.css. (Used to show off how you can use this data we are collecting.)

###### Folders:
- src (Has scripts that is used under the main program.)
- config (Inside this you have, conf.json. Make sure not to mess too much in here as it's used heavily in our script.)
- output (Everything our script creates goes here.)
- input (Every thing that our script is here. Some datasets have been removed from here, since they're too large to transfer.)

### Usage
to run this program type `$ py main.py`, or alternativily type `$ py3 main.py`. When using this you have to have your own emails with the file extension .eml, inside input/emails/.

### Using this to spread information
We are collecting data depending on how many spam, or ham emails we've found that day. Inside your output folder, you will have a file named statistics.json and a file named links.json.

##### Statistics.json example:

```json
{
2022: {
	14: {
		12: {
				"Spam": 999999,
				"Ham": 60000
		}
	}
}
}
```
We are sorting the contents is listed into different dates.

##### Links.json example:

```json
{
  "links": [
  "google.com",
  "gmail.com",
  "virustotal.com"
  ]
}
```

*NOTE: You can use this information whenever you want. This is fully accessable and should be used to make others warning that spam emails does exist, and they need to be carefull with what they click on.*
