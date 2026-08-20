from utils.colors import ColorRevolver


def test_color_revolver():
    colors = [
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    cr = ColorRevolver()

    assert cr.idx == 0

    for i in range(12):
        idx = i % 10
        color = cr.get_color()
        assert color == colors[idx]
        assert cr.idx == (idx + 1) % 10
