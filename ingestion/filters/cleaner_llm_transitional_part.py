from ollama import ChatResponse, Message, chat

import sys_config


def _generate_system_prompt() -> str:
    system_prompt = """
    Your task is to remove any introductory or navigational sentences that reference "this chapter", "the book", "skip to",
    or reader instructions. Keep only objective technical content. Output the cleaned text directly without explanation.
    """
    return system_prompt


#  in first line. Answer the refined content in the second line
def _generate_user_prompt(text: str) -> str:
    user_prompt = f"""
    ## Text
    {text}
    """

    return user_prompt


def remove_transitional_part(text: str):
    system_msg = Message(
        role="system",
        content=_generate_system_prompt(),
    )
    user_msg = Message(
        role="user",
        content=_generate_user_prompt(text),
    )

    res: ChatResponse = chat(
        model=sys_config.OLLAMA_GEMMA_MODEL_4_E2B,
        messages=[system_msg, user_msg],
    )

    res_content = res.message.content
    if res_content is not None:
        return res_content
    else:
        return ""


if __name__ == "__main__":
    text = """
    The term **responsive web design** was coined by Ethan Marcotte. In his seminal List Apart article (<a href="http://www.alistapart.com/articles/responsive-web-design/">http://www.alistapart.com/articles/responsive-web-design/</a>) he consolidated three existing techniques (flexible grid layout, flexible images, and media and media queries) into one unified approach and named it responsive web design. The term is often used to infer the same meaning as a number of other descriptions such as fluid design, elastic layout, rubber layout, liquid design, adaptive layout, cross-device design, and flexible design.
    To name just a few! However, as Mr. Marcotte and others have eloquently argued, a truly responsive methodology is actually more than merely altering the layout of a site based upon viewport sizes. Instead, it is to invert our entire current approach to web design. Instead of beginning with a fixed width desktop site design and scaling it down and re-flowing the content for smaller viewports, we should design for the smallest viewport first and then progressively enhance the design and content for larger viewports.
    """
    text_2 = """
    Until relatively recently, websites could be built at a fixed width, such as 960 pixels, with the expectation that all end users would get a fairly consistent experience. This fixed width wasn't too wide for laptop screens, and users with large resolution monitors merely had an abundance of margin either side.
    But now, there are smart phones. Apple's iPhone ushered in the first truly usable phone browsing experience, and many others have now followed that lead. Unlike the small-screen web browsing implementations of yesterday, that required the thumb dexterity of a Tiddlywinks world champion to use, people are now comfortably using their phones to browse the Web. In addition, there is a growing consumer trend of using small screen devices (tablets and netbooks, for example) in preference to their full screen brethren for content consumption in the home. The indisputable fact is that the number of people using these smaller screen devices to view the Internet is growing at an ever-increasing rate, whilst at the other end of the scale, 27 and 30 inch displays are now also commonplace. There is now a greater difference between the smallest screens browsing the Web and the largest than ever before.
    Thankfully, there is a solution to this ever-expanding browser and device landscape. A responsive web design, built with HTML5 and CSS3, allows a website to 'just work' across multiple devices and screens. And the best part is that the techniques are all implemented without the need for server based/backend solutions.
    In this chapter we shall:
    - Learn the importance of supporting small screen devices
    - Define "mobile website" design
    - Define "responsive website" design
    - Look at great examples of responsive web design
    - Learn the difference between viewport and screen sizes
    - Install and use viewport changing browser extensions
    - Use HTML5 to create cleaner and leaner markup
    - Use CSS3 to solve common design challenges

    """
    text_3 = """
    Since its release in 1995, JavaScript has gone through many changes. At first, it made adding interactive elements to web pages much simpler. Then it got more robust with DHTML and AJAX. Now, with Node.js, JavaScript has become a language that is used to build full-stack applications. The committee that is and has been in charge of shep‐ herding the changes to JavaScript is the European Computer Manufacturers Associa‐ tion (ECMA).
    Changes to the language are community-driven. They originate from proposals that community members write. Anyone [can submit a proposal](https://tc39.github.io/process-document/) to the ECMA committee. The responsibility of the ECMA committee is to manage and prioritize these propos‐ als in order to decide what is included in each spec. Proposals are taken through clearly defined stages, from stage 0, which represents the newest proposals, up through stage 4, which represents the finished proposals.
    The most recent major update to the specification was approved in June 2015<sup>1</sup> and is called by many names: ECMAScript 6, ES6, ES2015, and ES6Harmony. Based on cur‐ rent plans, new specs will be released on a yearly cycle. The 2016 release was relatively small, but it already looks like ES2017 will include quite a few useful features. We'll be using many of these new features in the book and will opt to use emerging JavaScript whenever possible.
    Many of these features are already supported by the newest browsers. We will also be covering how to convert your code from emerging JavaScript syntax to ES5 syntax that will work today in almost all browsers. The [kangax compatibility table](http://kangax.github.io/compat-table/esnext/) is a great place to stay informed about the latest JavaScript features and their varying degrees of support by browsers.
    <sup>1</sup> Abel Avram, ["ECMAScript 2015 Has Been Approved",](http://bit.ly/2nvMJjJ) InfoQ, June 17, 2015.
    In this chapter, we will show you all of the emerging JavaScript that we'll be using throughout the book. If you haven't made the switch to the latest syntax yet, now would be a good time to get started. If you are already comfortable with ES.Next lan‐ guage features, skip to the next chapter.
    """

    text_4 = """
    It's a cliche at this point to talk about [JavaScript Fatigue,](http://bit.ly/2pSiuE4) but the source of this fake illness can be traced back to the building process. In the past, you just added Java‐ Script files to your page. Now the JavaScript file has to be built, usually with an auto‐mated continuous delivery process. There's emerging syntax that has to be transpiled to work in all browsers. There's JSX that has to be converted to JavaScript. There's SCSS that you might want to preprocess. These components need to be tested, and they have to pass. You might love React, but now you also need to be a webpack expert, handling code splitting, compression, testing, and on and on.
    """

    print(remove_transitional_part(text_4))
