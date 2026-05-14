from sentence_transformers import CrossEncoder

# cross-encoder/ms-marco-MiniLM-L-12-v2
# cross-encoder/ms-marco-MiniLM-L6-v2
# cross-encoder/ms-marco-TinyBERT-L6
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

test_c = [
    (
        "How many people live in Berlin?",
        "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    ),
    ("How many people live in Berlin?", "Berlin is well known for its museums."),
]

q = """What are the three fundamental tenets that constitute the core of responsive web design?"""
chunks = [
    """
    The term responsive web design was coined by Ethan Marcotte. In his seminal List Apart article (<a href="http://www.alistapart.com/articles/responsive-web-design/">http://www.alistapart.com/articles/responsive-web-design/</a>) he consolidated three existing techniques (flexible grid layout, flexible images, and media and media queries) into one unified approach and named it responsive web design. The term is often used to infer the same meaning as a number of other descriptions such as fluid design, elastic layout, rubber layout, liquid design, adaptive layout, cross-device design, and flexible design. To name just a few! However, as Mr. Marcotte and others have eloquently argued, a truly responsive methodology is actually more than merely altering the layout of a site based upon viewport sizes. Instead, it is to invert our entire current approach to web design. Instead of beginning with a fixed width desktop site design and scaling it down and re-flowing the content for smaller viewports, we should design for the smallest viewport first and then progressively enhance the design and content for larger viewports.
    """,
    """
    The polyfill that I generally use to add media query support to older versions of Internet Explorer only adds support for min/max-width media queries. There are more substantial media query polyfills that add a greater range of media query support but for a responsive design, **Respond.js** by Scott Jehl is simple to use, fast, and has
    always served me well.
    Respond.js (<a href="https://github.com/scottjehl/Respond">https://github.com/scottjehl/Respond</a>) can actually be used without Modernizr—just add it to the page in question, and as the author Scott Jehl himself says, "Crack open Internet Explorer and pump fists in delight".
    So, before we integrate Respond.js with Modernizer, let's do just that. Drop Respond.js straight into our page (just add it after the Modernizr file we already added) and check it does what we want for IE. To do this, download the file, save it in a suitable location, and link to it in the <head> section:
    """,
    """
    While we could simply focus on how to create web pages and websites, none of this is possible without the underlying hardware and software components that support the pages we create. Examining what these components are and how they interact helps us understand what our server is capable of. The diagram below represents the basic elements of a web server. Hardware, an operating system, and an http server comprise the bare necessities. The addition of a database and scripting language extend a server's capabilities and are utilized in most servers as well. Figure 6 Web Server Software Structure
    """,
    """
    In the last two chapters we looked at some of the new features and functionality that CSS3 provides. However, until now, everything we have looked at has been static. But CSS3 can do more.
    At present, chances are, if you need to animate elements on a web page you'll either write your own JavaScript to perform the required action or turn to a popular JavaScript library like jQuery to do the heavy lifting. However, someone involved with CSS3 clearly has issues with JavaScript's ubiquity in this area and they're looking to encroach on JavaScript's dominance. While CSS3 isn't likely to usurp jQuery or the like anytime soon, it's perfectly capable of things like smoothing transitions (for example, on mouse hover) and moving elements around the screen. This is great news for us, as it means for the growing number of devices sporting modern browsers (recent smart phones for example), we can use CSS to provide animations rather than relying on JavaScript. The upshot: you can probably scratch 'learn how to animate elements with jQuery' off the 'to do' list as we can now do all that fun stuff in pure CSS.
    """,
]

q_1 = """
 and how does each component function when a browser requests a webpage?
"""
chunks_book_1 = [
    """
Seeing as most of us would have a hard time remembering what IP address is needed to get to, say, Facebook (173.252.100.16) or the Weather Channel (96.8.80.132) we instead use URLs, universal resource locators. This allows us to use www.facebook.com and www. weather.com to get to where we want to go without referring to a long list of IP addresses. Specialized servers (called name servers) all around the world are responsible for responding to requests from computers for this information
    """,
    """
    While we could simply focus on how to create web pages and websites, none of this is possible without the underlying hardware and software components that support the pages we create. Examining what these components are and how they interact helps us understand what our server is capable of. The diagram below represents the basic elements of a web server. Hardware, an operating system, and an http server comprise the bare necessities. The addition of a database and scripting language extend a server's capabilities and are utilized in most servers as well. Figure 6 Web Server Software Structure

    """,
    """
The polyfill that I generally use to add media query support to older versions of Internet Explorer
    """,
    """hi""",
]

if __name__ == "__main__":
    # print(len(chunks_book_1[0]))

    test_x = []
    for c in chunks_book_1:
        test_x.append((q_1, c))
    scores = model.predict(test_x)
    print(scores)
    for i, s in enumerate(scores):
        fs = float(s)
        if fs > 0:
            print(fs)
            # print(test_x[i][1])
