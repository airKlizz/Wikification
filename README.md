# Wikification

This objective is to retrieve wikify a text using the token classification Bert model from Hugging Face. 

## Dataset

I create a dataset composed of wikipedia passages with hyperlinks to others wikipedia articles. 

Example of a passage :

```
Patrick Lavon Mahomes II (born September 17, 1995) is an  <a>American football</a> <a>quarterback</a> for the <a>Kansas City Chiefs</a> of the <a>National Football League</a> (NFL). He is the son of former <a>Major League Baseball</a> (MLB) pitcher <a>Pat Mahomes</a>. He initially played <a>college football</a> and college <a>baseball</a> at <a>Texas Tech University</a>. Following his sophomore year, he quit baseball to focus solely on football. In his junior year, he led all <a>NCAA Division I FBS</a> players in multiple categories including passing yards (5,052 yards) and passing touchdowns (53 touchdowns). He then entered the <a>2017 NFL Draft</a> where he was the tenth overall selection by the Kansas City Chiefs.
```

*<a> </a> correspond to hyperlink*

