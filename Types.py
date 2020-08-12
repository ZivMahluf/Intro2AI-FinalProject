from typing import List, Tuple, Union


NumberType = Union[int, float]

CardType = Tuple[int, int]

CardListType = List[CardType]

# attacking cards, defending cards
FieldType = Tuple[CardListType, CardListType]

# attacking cards, defending cards, number of cards in deck, number of cards in each player's hand
TableType = Tuple[CardListType, CardListType, int, List[int]]

# attacking cards, defending cards, legal attack cards, legal defence cards, number of cards in deck, number of cards in each player's hand
StateType = Tuple[CardListType, CardListType, CardListType, CardListType, int, List[int]]
