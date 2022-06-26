from queue import Queue
import cv2
import numpy as np
from skimage.morphology import reconstruction
from skimage import img_as_float, img_as_ubyte


def tup(p):
    return int(p[0]), int(p[1])


# lab5 -- (5,10), (450,577) za 0.7 scale
# SETUP
img = cv2.imread('./labirint-slike/labb.png')
find_ulaz_izlaz = True
ulaz_izlaz_boja = [[0, 0, 240], [100, 100, 255]]
velicina_ulaz_izlaz = 3
scale = 0.6  # Ovisno od kompleksnosti labirinta
disable_erosion = False
add_boundaries = True
# -----------------------------------------------------------

# promijena dimenzije slike operacijom cv2.resize
h, w = img.shape[:2]
h = int(h * scale)
w = int(w * scale)
img = cv2.resize(img, (w, h))
copy = np.copy(img)
copy_2 = np.copy(img)

# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.imshow("Original", img)

# Bluranje slike i konverzija u sivoskaliranu
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray, (3, 3), 0)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 2)
mask = cv2.inRange(gray, 100, 255)

# cv2.namedWindow("Sivoskalirana", cv2.WINDOW_NORMAL)
# cv2.imshow("Sivoskalirana", gray)
# cv2.namedWindow("Adaptivni prag", cv2.WINDOW_NORMAL)
# cv2.imshow("Adaptivni prag", thresh)
# cv2.waitKey(0)

# Marker slika
seed = np.zeros_like(gray)
size = 40
seed[h // 2 - size:h // 2 + size, w // 2 - size:w // 2 + size] = gray[h // 2 - size:h // 2 + size,
                                                                 w // 2 - size:w // 2 + size]
# ---------- https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html,
# https://stackoverflow.com/questions/67051794/detect-maze-location-on-an-image
rec_1 = reconstruction(img_as_float(seed), img_as_float(gray))
rec_1[rec_1 < 0.50] = 0.
rec_1[rec_1 >= 0.50] = 1.  # Promijeniti po potrebi

# https://stackoverflow.com/questions/29104091/morphological-reconstruction-in-opencv
# slika = cv2.dilate(gray, seed, iterations=1)

# cv2.namedWindow("Rezultat dilatacije", cv2.WINDOW_NORMAL)
# cv2.imshow("Rezultat dilatacije",rec_1)
# cv2.waitKey(0)

seed_2 = np.ones_like(rec_1)
size_2 = 200
seed_2[h // 2 - size_2:h // 2 + size_2, w // 2 - size_2:w // 2 + size_2] = rec_1[h // 2 - size_2:h // 2 + size_2,
                                                                           w // 2 - size_2:w // 2 + size_2]
rec_2 = reconstruction(seed_2, rec_1, method='erosion', footprint=np.ones((11, 11)))
if disable_erosion:
    rec_2 = rec_1

# cv2.namedWindow("Rezultat erozije", cv2.WINDOW_NORMAL)
# cv2.imshow("Rezultat erozije", img_as_ubyte(rec_2))
# cv2.waitKey(0)
# ----------

# Konverzija natrag u sliku prepoznatljivu OpenCV biblioteci
gray = img_as_ubyte(rec_2)
mask = cv2.inRange(gray, 100, 255)

# Nalaženje ivica
# mask = cv2.bitwise_not(mask)
kernel = np.ones((111, 111), np.uint8)
morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
morph = 255 - morph

if np.mean(morph) == 255:
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((111, 111), np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    morph = 255 - morph

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def find_corners(image, height, width):  # Convex hull
    found_corners = [[[0, 0], 0] for _ in range(4)]
    for y in range(height):
        for x in range(width):
            if image[y][x] == 255:
                scores = [(height - y) + (width - x), (height - y) + x, y + x, y + (width - x)]
                for a in range(len(scores)):
                    if scores[a] > found_corners[a][1]:
                        found_corners[a][1] = scores[a]
                        found_corners[a][0] = [x, y]
    return found_corners


# https://stackoverflow.com/questions/40203932/drawing-a-rectangle-around-all-contours-in-opencv-python
# https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/
boxes = []
approx = None
for c in contours:
    (img_x, img_y, img_w, img_h) = cv2.boundingRect(c)
    boxes.append([img_x, img_y, img_x + img_w, img_y + img_h])
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)

# Pronalazak objekta (4 - kvadrat, 3 - trougao) na osnovu broja ivica
print("Aproksimacija:", len(approx))
if len(approx) == 4:
    # morph = np.bitwise_not(morph)
    corners = find_corners(morph, h, w)
    corners = [corner[0] for corner in corners]
else:
    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]
    corners = [[left, top], [right, top], [right, bottom], [left, bottom]]

cv2.drawContours(gray, contours, 0, (0, 0, 0), 2)
# cv2.namedWindow("Pronalazak kvadratnog podrucja", cv2.WINDOW_NORMAL)
# cv2.imshow("Pronalazak kvadratnog podrucja", gray)
# cv2.waitKey(0)

# Crtanje linija
img_borders = np.copy(img)
img_borders = cv2.cvtColor(img_borders, cv2.COLOR_GRAY2BGR)
for a in range(len(corners)):
    prev = corners[a - 1]
    curr = corners[a]
    cv2.line(img_borders, tup(prev), tup(curr), (0, 200, 0), 2)

# cv2.namedWindow("Granice", cv2.WINDOW_NORMAL)
# cv2.imshow("Granice", img_borders)
# cv2.waitKey(0)

# Crtanje ivica
# for corner in corners:
#  cv2.circle(img, tup(corner[0]), 4, (255, 255, 0), -1)

# Perspektivna transformacija
rectify = np.array([[0, 0], [w, 0], [w, h], [0, h]])
corners = np.array(corners)
H, _ = cv2.findHomography(corners, rectify)
if add_boundaries:
    cv2.drawContours(copy_2, contours, 0, (0, 0, 0), 2)
warped_img = cv2.warpPerspective(copy_2, H, (w, h))

# cv2.namedWindow("Perspektivna projekcija", cv2.WINDOW_NORMAL)
# cv2.imshow("Perspektivna projekcija", warped_img)
# cv2.waitKey(0)

# Nalazenje krugova za ulaz i izlaz
red_points_mask = cv2.inRange(warped_img, np.array(ulaz_izlaz_boja[0]), np.array(ulaz_izlaz_boja[1]))
entrance_coordinates = cv2.findNonZero(red_points_mask)
# CCL algoritam koji pronalazi povezanost izmedju "blob-like" regija (filtriranje blobova na slici)
entrance_values = cv2.connectedComponentsWithStats(red_points_mask, 2, cv2.CV_32S)
centroids = entrance_values[3][1:]
warped_img[red_points_mask > 0] = [255, 255, 255]

if find_ulaz_izlaz:
    cv2.circle(warped_img, tup(centroids[0]), velicina_ulaz_izlaz, (0, 255, 0), -1)
    cv2.circle(warped_img, tup(centroids[1]), velicina_ulaz_izlaz, (0, 255, 0), -1)

# cv2.namedWindow("Ulaz i izlaz", cv2.WINDOW_NORMAL)
# cv2.imshow("Ulaz i izlaz", warped_img)
# cv2.waitKey(0)

if len(centroids) < 2 and find_ulaz_izlaz:
    raise Exception("Ulaz i izlaz labirinta nije pronadjen")
elif not find_ulaz_izlaz:
    start_coord = tup([20, 20])
    end_coord = tup([w - 20, h - 20])
else:
    start_coord = tup(centroids[0])
    end_coord = tup(centroids[1])


# ------------------------------------------------------------------------------------------------------------------

# Maze Solver Djikstra

class Vertex:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord
        self.d = float('inf')  # distance from source
        self.parent_x = None
        self.parent_y = None
        self.processed = False
        self.index_in_queue = None


def get_neighbors(mat, r, c):
    shape = mat.shape
    neighbors = []
    # ensure neighbors are within image boundaries
    if r > 0 and not mat[r - 1][c].processed:
        neighbors.append(mat[r - 1][c])
    if r < shape[0] - 1 and not mat[r + 1][c].processed:
        neighbors.append(mat[r + 1][c])
    if c > 0 and not mat[r][c - 1].processed:
        neighbors.append(mat[r][c - 1])
    if c < shape[1] - 1 and not mat[r][c + 1].processed:
        neighbors.append(mat[r][c + 1])
    return neighbors


def bubble_up(queue, index):
    if index <= 0:
        return queue
    p_index = (index - 1) // 2
    if queue[index].d < queue[p_index].d:
        queue[index], queue[p_index] = queue[p_index], queue[index]
        queue[index].index_in_queue = index
        queue[p_index].index_in_queue = p_index
        queue = bubble_up(queue, p_index)
    return queue


def bubble_down(queue, index):
    length = len(queue)
    lc_index = 2 * index + 1
    rc_index = lc_index + 1
    if lc_index >= length:
        return queue
    if lc_index < length <= rc_index:  # just left child
        if queue[index].d > queue[lc_index].d:
            queue[index], queue[lc_index] = queue[lc_index], queue[index]
            queue[index].index_in_queue = index
            queue[lc_index].index_in_queue = lc_index
            queue = bubble_down(queue, lc_index)
    else:
        small = lc_index
        if queue[lc_index].d > queue[rc_index].d:
            small = rc_index
        if queue[small].d < queue[index].d:
            queue[index], queue[small] = queue[small], queue[index]
            queue[index].index_in_queue = index
            queue[small].index_in_queue = small
            queue = bubble_down(queue, small)
    return queue


def get_distance(img, u, v):
    return 0.1 + (float(img[v][0]) - float(img[u][0])) ** 2 + (float(img[v][1]) - float(img[u][1])) ** 2 + (
            float(img[v][2]) - float(img[u][2])) ** 2
    # return 0.1 + (float(img[v]) - float(img[u])) ** 2


def drawPath(img, path, thickness=2):
    x0, y0 = path[0]
    for vertex in path[1:]:
        x1, y1 = vertex
        cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), thickness)
        x0, y0 = vertex


def find_shortest_path(img, src, dst):
    pq = []  # min-heap priority queue
    source_x = src[0]
    source_y = src[1]
    dest_x = dst[0]
    dest_y = dst[1]
    imagerows, imagecols = img.shape[0], img.shape[1]
    matrix = np.full((imagerows, imagecols), None)  # access by matrix[row][col]
    for r in range(imagerows):
        for c in range(imagecols):
            matrix[r][c] = Vertex(c, r)
            matrix[r][c].index_in_queue = len(pq)
            pq.append(matrix[r][c])
    matrix[source_y][source_x].d = 0
    pq = bubble_up(pq, matrix[source_y][source_x].index_in_queue)
    while len(pq) > 0:
        u = pq[0]
        u.processed = True
        pq[0] = pq[-1]
        pq[0].index_in_queue = 0
        pq.pop()
        pq = bubble_down(pq, 0)
        neighbors = get_neighbors(matrix, u.y, u.x)
        for v in neighbors:
            dist = get_distance(img, (u.y, u.x), (v.y, v.x))
            if u.d + dist < v.d:
                v.d = u.d + dist
                v.parent_x = u.x
                v.parent_y = u.y
                idx = v.index_in_queue
                pq = bubble_down(pq, idx)
                pq = bubble_up(pq, idx)

    path = []
    iter_v = matrix[dest_y][dest_x]
    path.append((dest_x, dest_y))
    while iter_v.y != source_y or iter_v.x != source_x:
        path.append((iter_v.x, iter_v.y))
        iter_v = matrix[iter_v.parent_y][iter_v.parent_x]

    path.append((source_x, source_y))
    return path


def solve_maze_Djikstra(start_pos, end_pos, solve=False):
    gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)  # Pretvori u sivoskaliranu
    blur_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    slika = cv2.filter2D(blur_gray, -1, kernel)
    _, bw_img = cv2.threshold(slika, 127, 255, cv2.THRESH_BINARY)
    final = cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR)

    # final = copy_2
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', final)
    cv2.waitKey(0)

    if solve:
        path = find_shortest_path(final, start_pos, end_pos)
        pathed_image = np.zeros_like(img)
        path_thickness = 1
        drawPath(pathed_image, path, path_thickness)

        # Reverse
        final_img = cv2.warpPerspective(pathed_image, np.linalg.inv(H), (w, h))
        final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
        final_img[np.where((final_img != [0, 0, 0]).all(axis=2))] = [255, 0, 0]
        final_img[np.where((final_img == [0, 0, 0]).all(axis=2))] = copy[np.where((final_img == [0, 0, 0]).all(axis=2))]

        # Prikaz slika

        img_1 = np.concatenate((img, morph), axis=1)
        img_2 = np.concatenate((img_borders, warped_img), axis=1)
        img_3 = np.concatenate((cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR), img_2), axis=0)
        cv2.namedWindow("Koraci rjesenja", cv2.WINDOW_NORMAL)
        cv2.imshow("Koraci rjesenja", img_3)

        cv2.namedWindow("Rjesenje Labirinta Djikstra", cv2.WINDOW_NORMAL)
        cv2.imshow("Rjesenje Labirinta Djikstra", final_img)


# BFS algoritam za nalazenje najkraceg puta na slici
def get_adjacent(img, pos):
    x, y = pos
    adj = []
    if img[y][x + 1] == 1:
        adj.append((x + 1, y))
    if img[y][x - 1] == 1:
        adj.append((x - 1, y))
    if img[y + 1][x] == 1:
        adj.append((x, y + 1))
    if img[y - 1][x] == 1:
        adj.append((x, y - 1))
    return adj


def bfs(img, start_pos, end_pos):
    queue = Queue()
    queue.put([start_pos])

    while not queue.empty():
        path = queue.get()
        last_pos = path[-1]
        if last_pos == end_pos:
            return path
        for adjacent in get_adjacent(img, last_pos):
            if adjacent not in path:
                img[adjacent[1]][adjacent[0]] = 0
                new_path = list(path)
                new_path.append(adjacent)
                queue.put(new_path)
    return None


def solve_maze_BFS(start_pos, end_pos):
    gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    slika = cv2.filter2D(blur_gray, -1, img_kernel)
    _, bw_img = cv2.threshold(slika, 127, 255, cv2.THRESH_BINARY)

    # Izvacivanje krugova sa s like
    cv2.circle(bw_img, tup(centroids[0]), 15, (255, 255, 255), -1)
    cv2.circle(bw_img, tup(centroids[1]), 15, (255, 255, 255), -1)
    normalized_img = cv2.normalize(bw_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)  # Konverzija u BGR
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)  # Konverzija u sivu
    gray_img = cv2.convertScaleAbs(gray_img)  # Konverzija u int

    # Pronalazak kontura i podebljanje ivica
    cnts = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for corner in cnts:
        cv2.drawContours(normalized_img, [corner], -1, 0, thickness=1)

    path = bfs(normalized_img, start_pos, end_pos)

    pathed_image = np.zeros_like(img)
    path_thickness = 2
    drawPath(pathed_image, path, path_thickness)

    # Reverse
    final_img = cv2.warpPerspective(pathed_image, np.linalg.inv(H), (w, h))
    final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
    final_img[np.where((final_img != [0, 0, 0]).all(axis=2))] = [255, 0, 0]
    final_img[np.where((final_img == [0, 0, 0]).all(axis=2))] = copy[np.where((final_img == [0, 0, 0]).all(axis=2))]

    # Prikaz slika

    img_1 = np.concatenate((img, morph), axis=1)
    img_2 = np.concatenate((img_borders, warped_img), axis=1)
    img_3 = np.concatenate((cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR), img_2), axis=0)
    cv2.namedWindow("Koraci rjesenja", cv2.WINDOW_NORMAL)
    cv2.imshow("Koraci rjesenja", img_3)

    cv2.namedWindow("Rjesenje Labirinta BFS", cv2.WINDOW_NORMAL)
    cv2.imshow("Rjesenje Labirinta BFS", final_img)


solve_maze_Djikstra(start_coord, end_coord, True)
# solve_maze_BFS(start_coord, end_coord)
cv2.waitKey(0)
