import React, { useState, useEffect } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  CssBaseline,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useTheme,
  useMediaQuery,
  Avatar,
  Tooltip,
  Stack,
  CircularProgress,
  styled
} from '@mui/material';
import { 
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Chat as ChatIcon,
  Settings as SettingsIcon,
  Terminal as TerminalIcon,
  Close as CloseIcon,
  Minimize as MinimizeIcon,
  CropSquare as CropSquareIcon,
  FilterNone as FilterNoneIcon,
} from '@mui/icons-material';
import { useAppSelector } from '@/store/hooks';
import { selectIsLoading } from '@/features/app/appSlice';

const drawerWidth = 240;

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  WebkitAppRegion: 'drag',
  '& .MuiToolbar-root': {
    minHeight: '32px',
    paddingLeft: theme.spacing(2),
    paddingRight: theme.spacing(2),
  },
}));

const StyledToolbar = styled(Toolbar)(() => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
}));

const WindowControls = styled('div')({
  WebkitAppRegion: 'no-drag',
  display: 'flex',
  alignItems: 'center',
  '& button': {
    marginLeft: '1px',
    padding: '4px',
    '&:hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.1)',
    },
  },
});

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })<{
  open?: boolean;
  isMobile: boolean;
}>(({ theme, open, isMobile }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  marginLeft: isMobile ? 0 : `-${drawerWidth}px`,
  ...(open && !isMobile && {
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: 0,
  }),
  height: 'calc(100vh - 32px)',
  overflow: 'auto',
  paddingBottom: theme.spacing(10),
}));

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Chat', icon: <ChatIcon />, path: '/chat' },
  { text: 'Terminal', icon: <TerminalIcon />, path: '/terminal' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];

export const Layout = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const isLoading = useAppSelector(selectIsLoading);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  };

  const handleMinimize = () => {
    if (window.electron?.window) {
      window.electron.window.minimize();
    }
  };

  const handleMaximize = () => {
    if (window.electron?.window) {
      window.electron.window.maximize();
      setIsMaximized(!isMaximized);
    }
  };

  const handleClose = () => {
    if (window.electron?.window) {
      window.electron.window.close();
    }
  };

  useEffect(() => {
    const handleMaximized = () => setIsMaximized(true);
    const handleUnmaximized = () => setIsMaximized(false);

    const checkMaximized = async () => {
      if (window.electron?.window?.isMaximized) {
        const maximized = await window.electron.window.isMaximized();
        setIsMaximized(maximized);
      }
    };

    checkMaximized();
    
    if (window.electron) {
      window.electron.on('maximized', handleMaximized);
      window.electron.on('unmaximized', handleUnmaximized);
    }

    return () => {
      if (window.electron) {
        window.electron.off('maximized', handleMaximized);
        window.electron.off('unmaximized', handleUnmaximized);
      }
    };
  }, []);

  const drawer = (
    <div>
      <Toolbar>
        <Box display="flex" alignItems="center" width="100%" justifyContent="space-between">
          <Box display="flex" alignItems="center">
            <Avatar
              src="/logo192.png"
              alt="JARVIS"
              sx={{ width: 32, height: 32, marginRight: 1 }}
            />
            <Typography variant="h6" noWrap component="div">
              JARVIS
            </Typography>
          </Box>
          {isMobile && (
            <IconButton onClick={handleDrawerToggle}>
              <CloseIcon />
            </IconButton>
          )}
        </Box>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => handleNavigation(item.path)}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <CssBaseline />
      <StyledAppBar position="fixed" elevation={0}>
        <StyledToolbar>
          <Box display="flex" alignItems="center">
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, WebkitAppRegion: 'no-drag' }}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div">
              {menuItems.find((item) => item.path === location.pathname)?.text || 'JARVIS'}
            </Typography>
          </Box>
          <WindowControls>
            <Tooltip title="Minimize">
              <IconButton size="small" onClick={handleMinimize}>
                <MinimizeIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title={isMaximized ? 'Restore' : 'Maximize'}>
              <IconButton size="small" onClick={handleMaximize}>
                {isMaximized ? <FilterNoneIcon fontSize="small" /> : <CropSquareIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Close">
              <IconButton size="small" onClick={handleClose} sx={{ '&:hover': { color: '#ff3e41' } }}>
                <CloseIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </WindowControls>
        </StyledToolbar>
      </StyledAppBar>
      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
        aria-label="mailbox folders"
      >
        <Drawer
          variant={isMobile ? 'temporary' : 'persistent'}
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
            },
          }}
        >
          {drawer}
        </Drawer>
      </Box>
      <Main open={mobileOpen} isMobile={isMobile}>
        <Toolbar /> {/* This Toolbar pushes content below the AppBar */}
        {isLoading ? (
          <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            minHeight="calc(100vh - 200px)"
          >
            <Stack spacing={2} alignItems="center">
              <CircularProgress size={60} thickness={4} />
              <Typography variant="body1" color="textSecondary">
                Initializing JARVIS...
              </Typography>
            </Stack>
          </Box>
        ) : (
          <Outlet />
        )}
      </Main>
    </Box>
  );
};

export default Layout;
