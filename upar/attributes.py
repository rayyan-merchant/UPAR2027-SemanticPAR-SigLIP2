ATTRIBUTES = [
    'Age-Young','Age-Adult','Age-Old','Gender-Female',
    'Hair-Length-Short','Hair-Length-Long','Hair-Length-Bald','UpperBody-Length-Short',
    'UpperBody-Color-Black','UpperBody-Color-Blue','UpperBody-Color-Brown',
    'UpperBody-Color-Green','UpperBody-Color-Grey','UpperBody-Color-Orange',
    'UpperBody-Color-Pink','UpperBody-Color-Purple','UpperBody-Color-Red',
    'UpperBody-Color-White','UpperBody-Color-Yellow','UpperBody-Color-Other',
    'LowerBody-Length-Short','LowerBody-Color-Black','LowerBody-Color-Blue',
    'LowerBody-Color-Brown','LowerBody-Color-Green','LowerBody-Color-Grey',
    'LowerBody-Color-Orange','LowerBody-Color-Pink','LowerBody-Color-Purple',
    'LowerBody-Color-Red','LowerBody-Color-White','LowerBody-Color-Yellow',
    'LowerBody-Color-Other','LowerBody-Type-Trousers&Shorts','LowerBody-Type-Skirt&Dress',
    'Accessory-Backpack','Accessory-Bag','Accessory-Glasses-Normal',
    'Accessory-Glasses-Sun','Accessory-Hat',
]

def make_prompt(attribute):
    if attribute.startswith('UpperBody-Color-'):
        c=attribute.split('-')[-1].lower()
        return f'this is a photo of a person wearing {c} upper body clothing.'
    if attribute.startswith('LowerBody-Color-'):
        c=attribute.split('-')[-1].lower()
        return f'this is a photo of a person wearing {c} lower body clothing.'
    prompts={
        'Age-Young':'this is a photo of a young person.',
        'Age-Adult':'this is a photo of an adult person.',
        'Age-Old':'this is a photo of an older person.',
        'Gender-Female':'this is a photo of a female-presenting person.',
        'Hair-Length-Short':'this is a photo of a person with short hair.',
        'Hair-Length-Long':'this is a photo of a person with long hair.',
        'Hair-Length-Bald':'this is a photo of a bald person.',
        'UpperBody-Length-Short':'this is a photo of a person wearing a short-sleeved top.',
        'LowerBody-Length-Short':'this is a photo of a person wearing shorts or a short skirt.',
        'LowerBody-Type-Trousers&Shorts':'this is a photo of a person wearing trousers or shorts.',
        'LowerBody-Type-Skirt&Dress':'this is a photo of a person wearing a skirt or dress.',
        'Accessory-Backpack':'this is a photo of a person carrying a backpack.',
        'Accessory-Bag':'this is a photo of a person carrying a handbag or shoulder bag.',
        'Accessory-Glasses-Normal':'this is a photo of a person wearing eyeglasses.',
        'Accessory-Glasses-Sun':'this is a photo of a person wearing sunglasses.',
        'Accessory-Hat':'this is a photo of a person wearing a hat.',
    }
    return prompts[attribute]

PROMPTS=[make_prompt(a) for a in ATTRIBUTES]
