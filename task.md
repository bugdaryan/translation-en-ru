* Write code with a cli interface for a simple model for English-to-Russian translation. The code should have an option to train the model (so it's not just a model that's already pretrained), and some very basic inference (just an option to translate from English to Russian).
* Come up with a way to make the translator undertranslate or overtranslate a word.

Undertranslation: A word is left as it is, even though it should be changed in translation. For example, "I have an apple". It should be translated as "У меня есть яблоко". But if we undertranslate the word "apple", the result would be something like "У меня есть Apple" (like we mean Apple the company)
Overtranslation: A word is translated, even though it should be left as it is. Example: "I work at Apple". The correct translation is "Я работаю в Apple", but if we overtranslate the word "apple", the result would be something like "Я работаю в Яблоке"
Please don't go crazy with this task. 
The goal is to write a proof of concept, not something perfect and accounting for any edge case.