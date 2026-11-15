# Who XARVIS is for

XARVIS turns NASA's SPHEREx all-sky survey into something you can look at: pick a point on the sky and see every
SPHEREx image of it as a time-lapse, with each source named from the catalogs, anything unexplained flagged, and
any object trackable. This note defines who that is for and what each group can actually do with what is shown.
It is a working definition for the team, to decide what to build next and what to leave out.

## Primary audience: students, teachers and science-curious visitors

People who know a little astronomy, or none, and want to see real NASA data rather than an artist's impression.

| What XARVIS shows | What they can do with it |
|---|---|
| A star's position in every SPHEREx image over the survey | Watch a nearby star (Barnard's Star, the test targets) move against the background: proper motion, measured, not described |
| Colour from two wavelengths taken at the same instant | See that "redder" means cooler or dustier: compare a brown dwarf with an ordinary star |
| The 3-D viewer with SPHEREx's real orbit and where it can point | Understand why a telescope can't see everything at once (the Sun limit, Earth in the way) |
| Every object named from Gaia, CatWISE, Hipparcos and JPL | Learn what is actually in a patch of sky, and which catalog knows it |

**What they need from us:** plain-language labels, the loading tips, a working default (24 images), no jargon
without an explanation.

## Secondary audience: amateur astronomers and citizen scientists

People with some observing experience who want to contribute, or to check something they saw.

| What XARVIS shows | What they can do with it |
|---|---|
| NEW objects: sources in no catalog, classified as static, variable, transient, moving or single | Shortlist candidates for follow-up with their own telescope, or report them |
| A "moving" NEW object with a fitted path | Track it: XARVIS collects every SPHEREx image along the path. Compare with known minor planets before claiming a discovery |
| Known asteroids identified in the images (JPL SB Ident, Horizons) | Check that an asteroid was where JPL predicted, and when SPHEREx caught it |
| Brightness against the object's own spectrum, with BRIGHTNESS CHANGE flags | Spot variable stars and flares in the infrared |

**What they need from us:** reliable NEW flags (few false alarms), exportable positions and times, clear
"this could still be a known object" caveats.

## Tertiary audience: researchers who want a quick look before downloading

Astronomers who will use the SPHEREx data seriously, and want to know in a minute whether a target is worth it.

| What XARVIS shows | What they can do with it |
|---|---|
| Per-image flags at the target (TRANSIENT, OUTLIER, saturated, bad pixels) | Decide which images to trust before downloading them from IRSA |
| Signal-to-noise per image and wavelength | See whether the target is detected at all, and in which bands |
| Positions compared with Gaia DR3 (median offset) | Judge the astrometry for the field |
| The images themselves, aligned north-up | Catch blends, ghosts and nearby bright stars that would contaminate photometry |

**What they need from us:** the numbers behind the pictures (the summary JSON is already there; a CSV export would
help), and honesty about display-only processing (the colour and sharpening never touch the measurements).

## Who it is not for (yet)

- **Precision photometry or astrometry.** XARVIS is a quick look. The pipeline's measurements are careful, but the
  web runs use a sample of images (8 to 200) spread over the survey, not all of them.
- **Discovery claims.** A NEW object means "not in these catalogs at these limits", not "unknown to science".
- **Real-time alerts.** IRSA releases images up to 60 days after they are taken.

## The hackathon judges

For the demo, the judges are an audience of their own: they need to see, in one or two minutes, that the site uses
real NASA data, that it works end to end, and that a person in one of the groups above would come back to it. The
strongest demo path is a test target with visible motion (Barnard's Star), one NEW object clicked and tracked, and
the loading card's tips while the data arrives.

## What this suggests we build next

1. **Export:** positions, times and flags of the target and every NEW object as CSV (researchers, citizen scientists).
2. **Shareable links** that reopen a target and date (teachers, judges).
3. **A guided first run:** one click on a test target from the intro screen (students, judges).
4. **"Could this be known?" checks** for NEW objects: a link to the Minor Planet Center's checker for moving ones.
