
def isWordGuessed(secretWord, lettersGuessed):
    lengthSecret = 0
    for l in secretWord:
        if l in lettersGuessed :
            lengthSecret += 1
    if(lengthSecret == len(secretWord)):
        return True
    else:
        return False

def getGuessedWord(secretWord, lettersGuessed):
    guessWord =''
    for l in secretWord:
        if l in lettersGuessed:
            guessWord += l
        else:
            guessWord += '_ '
    return guessWord

def getAvailableLetters(lettersGuessed):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    available = ''
    for l in letters:
        if l not in lettersGuessed:
            available += l
    return available

def hangman(secretWord):
    print('Welcome to the game Hangman!\nI am thinking of a word that is', len(secretWord),'letters long.')
    mistakesMade = 0
    lettersGuessed =[]
    while(mistakesMade<8):
        print('You have',8-mistakesMade,'guesses left.')
        available=getAvailableLetters(lettersGuessed)
        print('Available letters:',available)
        guessLetter=input('Please guess a letter: ')
        lettersGuessed.append(guessLetter)
        if guessLetter in secretWord:
            print('Good guess:',getGuessedWord(secretWord, lettersGuessed))
            print('----------------------')
        else:
            print('Oops! That letter is not in my word:',getGuessedWord(secretWord, lettersGuessed))
            print('----------------------')
            mistakesMade += 1
        if isWordGuessed(secretWord, lettersGuessed):
            print('Congratulations, you won!')
            break
        elif(mistakesMade == 8):
            print('Sorry, you ran out of guesses. The word was else.')

secretWord = 'apple'
hangman(secretWord)
