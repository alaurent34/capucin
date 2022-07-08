class ContinuousStatusError(Exception):
    ''''Generic Error '''
    pass

class ContinuousStatusExceptedFormatError(ContinuousStatusError):
    '''Raised when using a method expecting class attribute to have been transform beforehand.'''
    pass