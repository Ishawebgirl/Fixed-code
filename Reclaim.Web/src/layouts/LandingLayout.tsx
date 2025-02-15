import { HelmetProvider } from 'react-helmet-async'

interface ILandingLayout {
  children: any
  id: string
  image: string
}

export const LandingLayout = ({ children, id, image }: ILandingLayout) => {
  return (
    <>
      <HelmetProvider>
        <title>Reclaim SIU</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </HelmetProvider>
      <div id={id} className="landing" style={{ backgroundImage: 'url(' + image + ')' }}>
        <main>{children}</main>
      </div>
    </>
  )
}
