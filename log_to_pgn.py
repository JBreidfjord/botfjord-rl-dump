def pgn_generator(move_stack):
    move_count = 1
    pgn_str = ""
    for i, move in enumerate(move_stack, start=1):
        if i % 2 != 0:
            pgn_str += str(move_count) + ". " + move + " "
        else:
            pgn_str += move + " "
            move_count += 1

    return pgn_str.strip()


if __name__ == "__main__":
    log = input("Insert move stack from log: ").strip("[]").replace("'", "")
    move_stack = log.split(", ")
    pgn_str = pgn_generator(move_stack)
    print("PGN:")
    print(pgn_str)
