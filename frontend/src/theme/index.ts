import { createTheme, responsiveFontSizes } from '@mui/material/styles';

declare module '@mui/material/styles' {
  interface Theme {
    customShadows: {
      primary: string;
      secondary: string;
      elevation1: string;
      elevation2: string;
      glow: string;
      innerGlow: string;
    };
    gradients: {
      primary: string;
      secondary: string;
      glass: string;
    };
  }
  interface ThemeOptions {
    customShadows?: {
      primary?: string;
      secondary?: string;
      elevation1?: string;
      elevation2?: string;
      glow?: string;
      innerGlow?: string;
    };
    gradients?: {
      primary?: string;
      secondary?: string;
      glass?: string;
    };
  }
}

// Base theme configuration
const baseTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00a8e8',
      light: '#5fd3ff',
      dark: '#007ab5',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#ff3e41',
      light: '#ff7678',
      dark: '#c30016',
      contrastText: '#ffffff',
    },
    background: {
      default: '#0a0e17',
      paper: '#121a29',
    },
    text: {
      primary: '#e0e0e0',
      secondary: '#a0a0a0',
      disabled: '#707070',
    },
    divider: 'rgba(255, 255, 255, 0.08)',
    action: {
      active: '#ffffff',
      hover: 'rgba(0, 168, 232, 0.08)',
      selected: 'rgba(0, 168, 232, 0.16)',
      disabled: 'rgba(255, 255, 255, 0.3)',
      disabledBackground: 'rgba(255, 255, 255, 0.12)',
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.2,
      letterSpacing: '-0.01562em',
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.2,
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
      lineHeight: 1.2,
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.2,
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.2,
    },
    h6: {
      fontWeight: 600,
      fontSize: '1rem',
      lineHeight: 1.2,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  customShadows: {
    primary: '0 0 20px rgba(0, 168, 232, 0.5)',
    secondary: '0 0 20px rgba(255, 62, 65, 0.5)',
    elevation1: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    elevation2: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    glow: '0 0 15px rgba(0, 168, 232, 0.7)',
    innerGlow: 'inset 0 0 10px rgba(0, 168, 232, 0.3)',
  },
  gradients: {
    primary: 'linear-gradient(135deg, #00a8e8 0%, #0077b6 100%)',
    secondary: 'linear-gradient(135deg, #ff3e41 0%, #c1121f 100%)',
    glass: 'linear-gradient(135deg, rgba(10, 14, 23, 0.8) 0%, rgba(18, 26, 41, 0.9) 100%)',
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 999,
          padding: '8px 24px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 0 15px rgba(0, 168, 232, 0.5)',
          },
          '&.Mui-disabled': {
            opacity: 0.5,
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #00a8e8 0%, #0077b6 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #0095d1 0%, #006494 100%)',
          },
        },
        containedSecondary: {
          background: 'linear-gradient(135deg, #ff3e41 0%, #c1121f 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #e63946 0%, #a4161a 100%)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(18, 26, 41, 0.7)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          '&:hover': {
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'rgba(10, 14, 23, 0.8)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          background: 'rgba(10, 14, 23, 0.9)',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
  },
});

// Add responsive font sizes
const theme = responsiveFontSizes(baseTheme);

export { theme };
