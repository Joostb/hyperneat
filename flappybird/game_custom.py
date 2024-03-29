import neat

import pygame
from numpy.random import choice

FPS = 200
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 160  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79


class Bird(pygame.sprite.Sprite):

    def __init__(self, displayScreen):

        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('./flappybird/assets/redbird.png')

        self.x = int(SCREENWIDTH * 0.2)
        self.y = SCREENHEIGHT * 0.5

        self.rect = self.image.get_rect()
        self.height = self.rect.height
        self.screen = displayScreen

        self.playerVelY = -9
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False

        self.display(self.x, self.y)

    def display(self, x, y):

        self.screen.blit(self.image, (x, y))
        self.rect.x, self.rect.y = x, y

    def move(self, input):

        if input is not None:
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = True

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

        self.y += min(self.playerVelY, SCREENHEIGHT - self.y - self.height)
        self.y = max(self.y, 0)
        self.display(self.x, self.y)


class PipeBlock(pygame.sprite.Sprite):

    def __init__(self, image, upper):

        pygame.sprite.Sprite.__init__(self)

        if not upper:
            self.image = pygame.image.load(image)
        else:
            self.image = pygame.transform.rotate(pygame.image.load(image), 180)

        self.rect = self.image.get_rect()


class Pipe(pygame.sprite.Sprite):

    def __init__(self, screen, x):
        pygame.sprite.Sprite.__init__(self)

        self.screen = screen
        self.lowerBlock = PipeBlock('./flappybird/assets/pipe-red.png', False)
        self.upperBlock = PipeBlock('./flappybird/assets/pipe-red.png', True)

        self.pipeWidth = self.upperBlock.rect.width
        self.x = x

        heights = self.getHeight()
        self.upperY, self.lowerY = heights[0], heights[1]

        self.behindBird = 0
        self.display()

    def getHeight(self):
        # randVal = randint(1,10)
        randVal = choice([1, 2, 3, 4, 5, 6, 7, 8, 9],
                         p=[0.04, 0.04 * 2, 0.04 * 3, 0.04 * 4, 0.04 * 5, 0.04 * 4, 0.04 * 3, 0.04 * 2, 0.04])

        midYPos = 106 + 30 * randVal

        upperPos = midYPos - (PIPEGAPSIZE / 2)
        lowerPos = midYPos + (PIPEGAPSIZE / 2)

        # print(upperPos)
        # print(lowerPos)
        # print('-------')
        return ([upperPos, lowerPos])

    def display(self):
        self.screen.blit(self.lowerBlock.image, (self.x, self.lowerY))
        self.screen.blit(self.upperBlock.image, (self.x, self.upperY - self.upperBlock.rect.height))
        self.upperBlock.rect.x, self.upperBlock.rect.y = self.x, (self.upperY - self.upperBlock.rect.height)
        self.lowerBlock.rect.x, self.lowerBlock.rect.y = self.x, self.lowerY

    def move(self):
        self.x -= 3

        if self.x <= 0:
            self.x = SCREENWIDTH
            heights = self.getHeight()
            self.upperY, self.lowerY = heights[0], heights[1]
            self.behindBird = 0

        self.display()
        return [self.x + (self.pipeWidth / 2), self.upperY, self.lowerY]


def game(genome, config, display_screen=True):
    BACKGROUND = pygame.image.load('./flappybird/assets/background.png') if display_screen else None

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()

    FPSCLOCK = pygame.time.Clock()
    DISPLAY = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    SCORE = 0

    bird = Bird(DISPLAY)
    pipe1 = Pipe(DISPLAY, SCREENWIDTH - 10)
    pipe2 = Pipe(DISPLAY, SCREENWIDTH - 10 + (SCREENWIDTH / 2))

    pipeGroup = pygame.sprite.Group()
    pipeGroup.add(pipe1.upperBlock)
    pipeGroup.add(pipe2.upperBlock)
    pipeGroup.add(pipe1.lowerBlock)
    pipeGroup.add(pipe2.lowerBlock)

    moved = False

    time = 0

    while True:
        if display_screen:
            DISPLAY.blit(BACKGROUND, (0, 0))

        if (pipe1.x < pipe2.x and pipe1.behindBird == 0) or (pipe2.x < pipe1.x and pipe2.behindBird == 1):
            input = (bird.y, pipe1.x, pipe1.upperY, pipe1.lowerY)
            centerY = (pipe1.upperY + pipe1.lowerY) / 2
        elif (pipe1.x < pipe2.x and pipe1.behindBird == 1) or (pipe2.x < pipe1.x and pipe2.behindBird == 0):
            input = (bird.y, pipe2.x, pipe2.upperY, pipe2.lowerY)
            centerY = (pipe2.upperY + pipe2.lowerY) / 2

        # print(input)
        vertDist = (((bird.y - centerY) ** 2) * 100) / (512 * 512)
        time += 1

        fitness = SCORE - vertDist + (time / 10.0)

        t = pygame.sprite.spritecollideany(bird, pipeGroup)

        if t is not None or (bird.y == 512 - bird.height) or (bird.y == 0):
            # print("GAME OVER")
            # print("FINAL SCORE IS %d"%fitness)
            return fitness, SCORE / 10

        output = net.activate(input)

        if output[0] >= 0.5:
            bird.move("UP")
            moved = True

        if not moved:
            bird.move(None)
        else:
            moved = False

        pipe1Pos = pipe1.move()
        if pipe1Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2):
            if pipe1.behindBird == 0:
                pipe1.behindBird = 1
                SCORE += 10
                # print("SCORE IS %d" % (SCORE / 10))

        pipe2Pos = pipe2.move()
        if pipe2Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2):
            if pipe2.behindBird == 0:
                pipe2.behindBird = 1
                SCORE += 10
                # print("SCORE IS %d" % (SCORE / 10))

        if display_screen:
            pygame.display.update()
        FPSCLOCK.tick(FPS)

# class FlappyGame:
#     def __init__(self, display_screen=True, genome=None, config=None):
#         self.display_screen = display_screen
#
#         self.BACKGROUND = pygame.image.load('./flappybird/assets/background.png') if display_screen else None
#
#         if genome is not None:
#             self.net = neat.nn.FeedForwardNetwork.create(genome, config)
#
#         pygame.init()
#
#         self.FPSCLOCK = pygame.time.Clock()
#         self.DISPLAY = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
#
#         pygame.display.set_caption('Flappy Bird')
#
#         self.SCORE = 0
#
#         self.bird = Bird(self.DISPLAY)
#         self.pipe1 = Pipe(self.DISPLAY, SCREENWIDTH - 10)
#         self.pipe2 = Pipe(self.DISPLAY, SCREENWIDTH - 10 + (SCREENWIDTH / 2))
#
#         self.pipeGroup = pygame.sprite.Group()
#         self.pipeGroup.add(self.pipe1.upperBlock)
#         self.pipeGroup.add(self.pipe2.upperBlock)
#         self.pipeGroup.add(self.pipe1.lowerBlock)
#         self.pipeGroup.add(self.pipe2.lowerBlock)
#
#         self.moved = False
#
#         self.time = 0
#
#     def get_state(self):
#         if self.display_screen:
#             self.DISPLAY.blit(self.BACKGROUND, (0, 0))
#
#         if (self.pipe1.x < self.pipe2.x and self.pipe1.behindBird == 0) or \
#                 (self.pipe2.x < self.pipe1.x and self.pipe2.behindBird == 1):
#             self.input = (self.bird.y, self.pipe1.x, self.pipe1.upperY, self.pipe1.lowerY)
#             self.centerY = (self.pipe1.upperY + self.pipe1.lowerY) / 2
#         elif (self.pipe1.x < self.pipe2.x and self.pipe1.behindBird == 1) or \
#                 (self.pipe2.x < self.pipe1.x and self.pipe2.behindBird == 0):
#             self.input = (self.bird.y, self.pipe2.x, self.pipe2.upperY, self.pipe2.lowerY)
#             self.centerY = (self.pipe2.upperY + self.pipe2.lowerY) / 2
#         return self.input
#
#     def act(self, Q):
#         # print(input)
#         vertDist = (((self.bird.y - self.centerY) ** 2) * 100) / (512 * 512)
#         self.time += 1
#
#         fitness = self.SCORE - vertDist + (self.time / 10.0)
#
#         t = pygame.sprite.spritecollideany(self.bird, self.pipeGroup)
#
#         if t is not None or (self.bird.y == 512 - self.bird.height) or (self.bird.y == 0):
#             # print("GAME OVER")
#             # print("FINAL SCORE IS %d"%fitness)
#             return fitness, self.SCORE / 10
#
#
#         if Q >= 0.5:
#             bird.move("UP")
#             moved = True
#
#         if not moved:
#             bird.move(None)
#         else:
#             moved = False
#
#         pipe1Pos = pipe1.move()
#         if pipe1Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2):
#             if pipe1.behindBird == 0:
#                 pipe1.behindBird = 1
#                 SCORE += 10
#                 # print("SCORE IS %d" % (SCORE / 10))
#
#         pipe2Pos = pipe2.move()
#         if pipe2Pos[0] <= int(SCREENWIDTH * 0.2) - int(bird.rect.width / 2):
#             if pipe2.behindBird == 0:
#                 pipe2.behindBird = 1
#                 SCORE += 10
#                 # print("SCORE IS %d" % (SCORE / 10))
#
#         if display_screen:
#             pygame.display.update()
#         FPSCLOCK.tick(FPS)
