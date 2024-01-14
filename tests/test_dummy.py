from tinyevals.dummy import dummy

def test_dummy():
    assert dummy() == 'dummy'
    assert dummy(1) == 2
    assert dummy('dummy') == 'dummy1'